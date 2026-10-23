#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
klasicna statisticka metoda Statsmodels Conditional KDE — Loto 7/39 — V1

Metoda:
    KDEMultivariateConditional
    Wang–Ryzin kernel za uređene diskretne promenljive.

Ulaz:
    sedam sortiranih brojeva prethodnog kola.

Izlaz:
    zajednička uslovna raspodela sedam sortiranih brojeva narednog kola.

Legalni prostor:
    1 <= y1 < y2 < ... < y7 <= 39.

Adaptacija:
    izvorna KDE raspodela uslovljava se na legalni prostor.
    Normalizacija se računa dinamičkim programiranjem.
    NEXT je globalni maksimum, ako ga pretraga potvrdi.

Nema:
    slučajnog uzorkovanja, mešanja redova, slučajnih podela,
    zaokruživanja, popravljanja duplikata ili kozmetičkih filtera.

Istorija:
    train 70%, validation 15%, test 15%, hronološki.
    Parametri se biraju samo validacijom.
    Test model koristi train + validation.
    Završni model koristi sve istorijske prelaze.
"""


import csv
import hashlib
import itertools
import math
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import statsmodels
    from numba import njit
    from statsmodels.nonparametric.kernel_density import (
        KDEMultivariateConditional,
    )
    from statsmodels.nonparametric import kernels
except ImportError as exc:
    raise SystemExit(
        "Potrebni su numpy, numba i statsmodels.\n"
        f"Nedostajuća zavisnost: {exc}"
    )


PATHS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4706_k84.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4706_k84_loto_2975.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4706_k84_loto_plus_1731.csv",
)

N = 39
K = 7

# Širine kernela: unapred zadati kandidati.
# Konačan par bira isključivo hronološka validacija.
INPUT_BANDWIDTHS = (0.35, 0.55, 0.75, 0.90, 0.97)
OUTPUT_BANDWIDTHS = (0.20, 0.40, 0.60, 0.80, 0.95)

BATCH = 64
MAP_SECONDS = 120.0
MAP_CHUNK_NODES = 256
LOG2 = math.log(2.0)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def fingerprint(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()[:16]


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        for line, fields in enumerate(csv.reader(stream), 1):
            if not fields or all(not value.strip() for value in fields):
                continue

            require(
                len(fields) == K,
                f"{path}, red {line}: očekivano sedam polja.",
            )
            try:
                values = [int(value.strip()) for value in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: neispravan ceo broj."
                ) from exc

            require(
                len(set(values)) == K
                and all(1 <= value <= N for value in values),
                f"{path}, red {line}: nelegalno kolo {values}.",
            )
            rows.append(sorted(values))

    require(len(rows) >= 100, f"{path}: potrebno najmanje 100 kola.")
    return np.asarray(rows, dtype=np.int64)


def scaled_kernel(distance, h):
    """
    Wang–Ryzin kernel podeljen zajedničkim faktorom (1-h):

        q(0) = 1
        q(d) = 0.5 * h**d, d > 0

    Zajednički faktor se tačno poništava pri normalizaciji
    preko legalnog prostora. Nije promena oblika kernela.
    """
    return np.where(distance == 0, 1.0, 0.5*np.power(h, distance))


def pair_distances(queries, centers):
    distance = np.zeros(
        (len(queries), len(centers)), dtype=np.int32
    )
    different = np.zeros_like(distance)
    for j in range(queries.shape[1]):
        delta = np.abs(queries[:, j, None] - centers[None, :, j])
        distance += delta
        different += delta != 0
    return distance, different


def log_product_kernel(queries, centers, h):
    distance, different = pair_distances(queries, centers)
    return distance*math.log(h) - different*LOG2


def context_weights(queries, centers, h):
    logs = log_product_kernel(queries, centers, h)
    logs -= logs.max(axis=1, keepdims=True)
    weights = np.exp(logs)
    weights /= weights.sum(axis=1, keepdims=True)
    require(np.all(np.isfinite(weights)), "Neispravne težine konteksta.")
    return weights


def kernel_table(targets, h, n):
    """
    table[position, number, historical_transition]
    """
    k = targets.shape[1]
    table = np.zeros((k, n+1, len(targets)), dtype=np.float64)
    numbers = np.arange(1, n+1)[:, None]
    for position in range(k):
        table[position, 1:] = scaled_kernel(
            np.abs(numbers-targets[None, :, position]), h
        )
    return table


def suffix_tables(table):
    """
    sums[t, a, i]:
        zbir proizvoda kernela za pozicije t,...,k-1,
        pri čemu prvi preostali broj mora biti >= a.

    maxima[t, a, i]:
        najveći takav proizvod za pojedinačnu KDE komponentu.

    Maxima daju gornju granicu za maksimum njihove mešavine.
    """
    k, width, count = table.shape
    n = width-1
    sums = np.zeros((k+1, n+2, count), dtype=np.float64)
    maxima = np.zeros_like(sums)
    sums[k, :, :] = 1.0
    maxima[k, :, :] = 1.0

    for t in range(k-1, -1, -1):
        for a in range(n, 0, -1):
            contribution = table[t, a]*sums[t+1, a+1]
            sums[t, a] = sums[t, a+1] + contribution

            product = table[t, a]*maxima[t+1, a+1]
            maxima[t, a] = np.maximum(maxima[t, a+1], product)

    require(
        np.all(np.isfinite(sums))
        and np.all(sums[0, 1] > 0),
        "Neispravna legalna normalizacija.",
    )
    return sums, maxima


def component_normalizers(targets, h, n):
    table = kernel_table(targets, h, n)
    # Za validaciju nije potrebno čuvati punu tabelu sufiksa.
    count = len(targets)
    following = np.ones((n+2, count), dtype=np.float64)

    for t in range(targets.shape[1]-1, -1, -1):
        current = np.zeros_like(following)
        for a in range(n, 0, -1):
            current[a] = (
                current[a+1] + table[t, a]*following[a+1]
            )
        following = current

    z = following[1]
    require(np.all(np.isfinite(z) & (z > 0)), "Neispravan Z komponente.")
    return z


def nll_values(train_x, train_y, query_x, query_y, hx, hy, n=N):
    normalizers = component_normalizers(train_y, hy, n)
    result = []

    for start in range(0, len(query_x), BATCH):
        x = query_x[start:start+BATCH]
        y = query_y[start:start+BATCH]

        weights = context_weights(x, train_x, hx)
        target_products = np.exp(log_product_kernel(y, train_y, hy))
        numerator = np.sum(weights*target_products, axis=1)
        denominator = weights @ normalizers

        require(
            np.all(np.isfinite(numerator) & (numerator > 0))
            and np.all(np.isfinite(denominator) & (denominator > 0)),
            "Nulta ili neispravna masa pri proveri NLL-a.",
        )

        probability = numerator/denominator
        require(np.all(probability <= 1+1e-9), "Masa cilja veća od 1.")
        result.extend((-np.log(probability)).tolist())

    return np.asarray(result)


def library_model(x, y, hx, hy):
    k = y.shape[1]
    return KDEMultivariateConditional(
        endog=y.astype(np.float64),
        exog=x.astype(np.float64),
        dep_type="o"*k,
        indep_type="o"*x.shape[1],
        bw=np.asarray([hy]*k + [hx]*x.shape[1], dtype=np.float64),
    )


def check_library(model, x, y, query_x, query_y, hx, hy):
    """
    Poređenje brzog računa sa stvarnim pdf() instaliranog statsmodels.

    Poredi se izvorna, još neuslovljena KDE masa.
    """
    weights = context_weights(query_x, x, hx)
    products = np.exp(log_product_kernel(query_y, y, hy))
    expected = np.sum(weights*products, axis=1)*(1-hy)**y.shape[1]

    actual = np.atleast_1d(model.pdf(
        endog_predict=query_y.astype(np.float64),
        exog_predict=query_x.astype(np.float64),
    ))

    require(np.all(np.isfinite(actual) & (actual > 0)),
            "Statsmodels je vratio neispravnu masu.")
    np.testing.assert_allclose(
        actual/expected, np.ones(len(expected)), atol=2e-8, rtol=2e-8
    )


def select_bandwidths(x, y, train_end, validation_end):
    train_x, train_y = x[:train_end], y[:train_end]
    val_x = x[train_end:validation_end]
    val_y = y[train_end:validation_end]

    best = None
    for hy in OUTPUT_BANDWIDTHS:
        normalizers = component_normalizers(train_y, hy, N)
        totals = np.zeros(len(INPUT_BANDWIDTHS), dtype=np.float64)

        for start in range(0, len(val_x), BATCH):
            bx = val_x[start:start+BATCH]
            by = val_y[start:start+BATCH]
            target_products = np.exp(
                log_product_kernel(by, train_y, hy)
            )

            distance, different = pair_distances(bx, train_x)
            for j, hx in enumerate(INPUT_BANDWIDTHS):
                logs = distance*math.log(hx)-different*LOG2
                logs -= logs.max(axis=1, keepdims=True)
                weights = np.exp(logs)
                weights /= weights.sum(axis=1, keepdims=True)

                numerator = np.sum(weights*target_products, axis=1)
                denominator = weights @ normalizers
                require(
                    np.all(numerator > 0) and np.all(denominator > 0),
                    "Nulta validaciona masa.",
                )
                totals[j] += np.sum(
                    np.log(denominator)-np.log(numerator)
                )

        for j, hx in enumerate(INPUT_BANDWIDTHS):
            score = float(totals[j]/len(val_x))
            require(np.isfinite(score), "Nefinitan validacioni NLL.")
            print(
                f"VALIDACIJA hx={hx:.2f}, hy={hy:.2f}, NLL={score:.8f}",
                flush=True,
            )
            if best is None or score < best["nll"]:
                best = {"hx": hx, "hy": hy, "nll": score}

    print(
        f"IZBOR: hx={best['hx']:.2f}, hy={best['hy']:.2f}, "
        f"validation NLL={best['nll']:.8f}",
        flush=True,
    )
    if (
        best["hx"] in (INPUT_BANDWIDTHS[0], INPUT_BANDWIDTHS[-1])
        or best["hy"] in (OUTPUT_BANDWIDTHS[0], OUTPUT_BANDWIDTHS[-1])
    ):
        print(
            "NAPOMENA: izabran je rub pretraženog raspona parametara.",
            flush=True,
        )
    return best


def mass_for_rows(rows, table, weights):
    result = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        product = weights.copy()
        for t, number in enumerate(row):
            product *= table[t, int(number)]
        result[i] = product.sum()
    return result


def combination_mask(row):
    mask = 0
    for number in row:
        mask |= 1 << (int(number)-1)
    return mask


def mask_combination(mask, n):
    return tuple(i+1 for i in range(n) if (int(mask) >> i) & 1)


@njit(cache=False)
def search_chunk(
    table, maxima, states, levels, lows, masks, bounds,
    top, best, best_mask, max_nodes
):
    """
    Nastavljiva dubinska pretraga.
    states sadrži težinu svake KDE komponente za trenutni prefiks.

    Gornja granica:
        sum_i(prefix_weight_i * best_component_suffix_i)

    To je >= najboljoj masi bilo kog zajedničkog nastavka.
    """
    k = table.shape[0]
    n = table.shape[1]-1
    count = table.shape[2]
    visited = 0

    while top > 0 and visited < max_nodes:
        top -= 1
        t = levels[top]
        low = lows[top]
        mask = masks[top]
        bound = bounds[top]
        prefix = states[top].copy()
        visited += 1

        tolerance = 1e-12*max(abs(bound), abs(best), 1e-300)
        if bound+tolerance < best:
            continue

        upper_number = n-(k-t)+1

        if t == k-1:
            for a in range(low, upper_number+1):
                value = 0.0
                for i in range(count):
                    value += prefix[i]*table[t, a, i]
                candidate_mask = mask | (np.int64(1) << (a-1))
                if value > best or (
                    value == best and candidate_mask < best_mask
                ):
                    best = value
                    best_mask = candidate_mask
            continue

        child_bounds = np.zeros(n+1, dtype=np.float64)
        for a in range(low, upper_number+1):
            value = 0.0
            for i in range(count):
                value += (
                    prefix[i]*table[t, a, i]*maxima[t+1, a+1, i]
                )
            child_bounds[a] = value

        # Manje granice se stavljaju prve:
        # veća granica biće sledeća skinuta sa steka.
        order = np.argsort(child_bounds)
        for index in range(len(order)):
            a = order[index]
            if a < low or a > upper_number:
                continue
            value = child_bounds[a]
            tolerance = 1e-12*max(abs(value), abs(best), 1e-300)
            if value+tolerance < best:
                continue

            if top >= states.shape[0]:
                raise RuntimeError("Nedovoljan kapacitet MAP steka.")

            for i in range(count):
                states[top, i] = prefix[i]*table[t, a, i]
            levels[top] = t+1
            lows[top] = a+1
            masks[top] = mask | (np.int64(1) << (a-1))
            bounds[top] = value
            top += 1

    return top, best, best_mask, visited


def exact_map(table, maxima, weights, seeds, seconds, verbose=True):
    k, width, count = table.shape
    n = width-1

    # Istorijske kombinacije služe samo kao početna donja granica.
    # Pretraga obuhvata i sve neistorijske legalne kombinacije.
    seed_values = mass_for_rows(seeds, table, weights)
    index = int(np.argmax(seed_values))
    best = float(seed_values[index])
    best_mask = np.int64(combination_mask(seeds[index]))

    capacity = 2+k*(n+1)
    states = np.zeros((capacity, count), dtype=np.float64)
    levels = np.zeros(capacity, dtype=np.int64)
    lows = np.zeros(capacity, dtype=np.int64)
    masks = np.zeros(capacity, dtype=np.int64)
    bounds = np.zeros(capacity, dtype=np.float64)

    states[0] = weights
    lows[0] = 1
    bounds[0] = float(weights @ maxima[0, 1])
    top = 1

    started = last_report = time.perf_counter()
    visited = 0

    while top:
        top, best, best_mask, done = search_chunk(
            table, maxima, states, levels, lows, masks, bounds,
            top, best, best_mask, MAP_CHUNK_NODES
        )
        visited += done
        now = time.perf_counter()

        if verbose and now-last_report >= 5:
            upper = max(best, float(bounds[:top].max())) if top else best
            print(
                f"  MAP: čvorovi={visited:,}, "
                f"gornja/donja granica={upper/best:.6g}, "
                f"vreme={now-started:.1f}s",
                flush=True,
            )
            last_report = now

        if top and now-started >= seconds:
            upper = max(best, float(bounds[:top].max()))
            raise RuntimeError(
                f"MAP nije potvrđen za {seconds:g}s. "
                f"Gornja/donja granica={upper/best:.8g}. "
                "NEXT se ne izdaje; kandidat nije zamena za potvrđen maksimum."
            )

    return mask_combination(best_mask, n), best, visited


def marginals_from_tables(table, sums, weights):
    k, width, count = table.shape
    n = width-1
    z = float(weights @ sums[0, 1])
    require(np.isfinite(z) and z > 0, "Neispravna normalizacija mešavine.")

    before = np.zeros((n+1, count), dtype=np.float64)
    before[1:] = 1.0
    marginal = np.zeros(n, dtype=np.float64)
    final_total = None

    for t in range(k):
        current = before*table[t]
        for a in range(1, n+1):
            marginal[a-1] += float(
                weights @ (current[a]*sums[t+1, a+1])
            )/z

        if t == k-1:
            final_total = float(weights @ current.sum(axis=0))

        cumulative = np.cumsum(current, axis=0)
        before = np.zeros_like(current)
        before[1:] = cumulative[:-1]

    np.testing.assert_allclose(final_total/z, 1.0, atol=2e-9, rtol=0)
    np.testing.assert_allclose(marginal.sum(), k, atol=2e-8, rtol=0)
    require(
        np.all(marginal >= -1e-10) and np.all(marginal <= 1+1e-9),
        "Neispravne marginalne mase.",
    )
    return z, marginal


def legal_rows(n, k):
    return np.asarray(
        list(itertools.combinations(range(1, n+1), k)),
        dtype=np.int64,
    )


def self_checks():
    print("Provere kernela, statsmodels PDF-a, normalizacije i MAP-a...",
          flush=True)

    h = 0.6
    centers = np.arange(1, 8, dtype=np.float64)
    actual = kernels.wang_ryzin(h, centers, 4.0)
    expected = (1-h)*scaled_kernel(np.abs(centers-4), h)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)

    # Sintetički primer, bez pristupa korisnikovim CSV-ovima.
    n, k = 6, 2
    rows = legal_rows(n, k)
    x = rows.copy()
    y = rows[::-1].copy()
    hx, hy = 0.55, 0.60
    query = rows[[3]]

    model = library_model(x, y, hx, hy)
    check_library(
        model, x, y,
        np.repeat(query, 3, axis=0), rows[[0, 5, 14]],
        hx, hy,
    )

    weights = context_weights(query, x, hx)[0]
    table = kernel_table(y, hy, n)
    sums, maxima = suffix_tables(table)
    z, marginals = marginals_from_tables(table, sums, weights)
    masses = mass_for_rows(rows, table, weights)
    p = masses/z

    np.testing.assert_allclose(p.sum(), 1.0, atol=2e-12)
    indicators = np.zeros((len(rows), n), dtype=np.float64)
    indicators[np.arange(len(rows))[:, None], rows-1] = 1
    np.testing.assert_allclose(p @ indicators, marginals, atol=2e-12)

    # Samo jedan početni kandidat: provera mora naći stvarni maksimum.
    numbers, maximum, _ = exact_map(
        table, maxima, weights, rows[:1], 30.0, verbose=False
    )
    np.testing.assert_allclose(maximum, masses.max(), atol=2e-12)
    require(len(numbers) == k and len(set(numbers)) == k,
            "Nelegalni MAP u proveri.")

    # Granični slučajevi: jedan broj i svi brojevi.
    for boundary_k in (1, n):
        candidates = legal_rows(n, boundary_k)
        targets = candidates.copy()
        w = np.full(len(targets), 1.0/len(targets))
        tab = kernel_table(targets, 0.4, n)
        ss, mm = suffix_tables(tab)
        zz, marginal = marginals_from_tables(tab, ss, w)
        values = mass_for_rows(candidates, tab, w)
        np.testing.assert_allclose(values.sum()/zz, 1.0, atol=2e-12)
        _, maximum, _ = exact_map(
            tab, mm, w, candidates[:1], 30.0, verbose=False
        )
        np.testing.assert_allclose(maximum, values.max(), atol=2e-12)
        np.testing.assert_allclose(marginal.sum(), boundary_k, atol=2e-12)

    print("Ugrađene provere prošle.", flush=True)


def longest_arithmetic(numbers):
    dp = [dict() for _ in numbers]
    best = ()
    for i in range(len(numbers)):
        for j in range(i):
            step = numbers[i]-numbers[j]
            candidate = dp[j].get(step, (numbers[j],))+(numbers[i],)
            if len(candidate) > len(dp[i].get(step, ())):
                dp[i][step] = candidate
            if len(candidate) > len(best):
                best = candidate
    return best


def diagnose(numbers, draws, weights, sums, z, marginal, sensitivity):
    messages = []
    sequence = longest_arithmetic(numbers)
    if len(sequence) >= 5:
        messages.append(f"Aritmetički podniz: {sequence}.")
    if all(number <= 19 for number in numbers):
        messages.append("Svih sedam brojeva je u 1–19.")
    if all(number >= 20 for number in numbers):
        messages.append("Svih sedam brojeva je u 20–39.")

    span = numbers[-1]-numbers[0]
    historical_spans = draws[:, -1]-draws[:, 0]
    if span <= float(np.quantile(historical_spans, 0.01)):
        messages.append("Raspon je u donjih 1% istorijskih raspona.")

    repeats = int(np.all(draws == np.asarray(numbers), axis=1).sum())
    if repeats:
        messages.append(
            f"Kombinacija postoji {repeats} puta u istoriji. "
            "KDE može imati maksimum na opaženoj kombinaciji."
        )

    # Težine nakon uslovljavanja na legalni prostor.
    legal_weights = weights*sums[0, 1]/z
    np.testing.assert_allclose(legal_weights.sum(), 1.0, atol=2e-10)
    effective = 1.0/float(np.sum(legal_weights**2))
    print(
        f"Efektivni broj istorijskih komponenti={effective:.2f}; "
        f"najveća težina={legal_weights.max():.6f}; "
        f"osetljivost marginala={sensitivity:.8g}",
        flush=True,
    )
    if effective < 5:
        messages.append("Raspodelom dominira mali broj istorijskih prelaza.")
    if sensitivity < 1e-6:
        messages.append("Slaba reakcija raspodele na proverene kontekste.")

    for message in messages:
        print("DIJAGNOSTIKA: "+message, flush=True)
    return messages


def process_file(path):
    print("\n"+"="*72, flush=True)
    print(Path(path).name, flush=True)

    draws = read_csv(path)
    x, y = draws[:-1], draws[1:]
    total = len(y)
    train_end = int(0.70*total)
    validation_end = int(0.85*total)

    print(
        f"Kola={len(draws)}; prelazi={total}; "
        f"train={train_end}; validation={validation_end-train_end}; "
        f"test={total-validation_end}.",
        flush=True,
    )

    best = select_bandwidths(x, y, train_end, validation_end)
    hx, hy = best["hx"], best["hy"]

    # Odvojeni test: model koristi samo ranije train+validation redove.
    test_x, test_y = x[validation_end:], y[validation_end:]
    trained_x, trained_y = x[:validation_end], y[:validation_end]
    test_model = library_model(trained_x, trained_y, hx, hy)
    probe_count = min(3, len(test_x))
    check_library(
        test_model, trained_x, trained_y,
        test_x[:probe_count], test_y[:probe_count], hx, hy,
    )

    test_nll = nll_values(
        trained_x, trained_y, test_x, test_y, hx, hy
    )
    print(f"ODVOJENI TEST: prosečni NLL={test_nll.mean():.8f}",
          flush=True)
    del test_model

    # KDE nema epohe: završno učenje čuva sve istorijske parove.
    final_model = library_model(x, y, hx, hy)
    require(final_model.nobs == total,
            "Završni model ne sadrži sve istorijske prelaze.")
    next_context = draws[-1:]
    print(f"ZAVRŠNI MODEL: svih {total} prelaza.", flush=True)

    weights = context_weights(next_context, x, hx)[0]
    table = kernel_table(y, hy, N)
    sums, maxima = suffix_tables(table)
    z, marginal = marginals_from_tables(table, sums, weights)

    # Provera izvorne KDE mase pre uslovljavanja.
    probes = y[[0, len(y)//2, len(y)-1]]
    check_library(
        final_model, x, y,
        np.repeat(next_context, len(probes), axis=0),
        probes, hx, hy,
    )

    # Ponavljanja su spojena samo za početne kandidate pretrage.
    # Istorijske komponente i njihove težine ostaju netaknute.
    seeds = np.unique(y, axis=0)
    print(
        f"MAP pretraga svih legalnih kombinacija; "
        f"limit={MAP_SECONDS:g}s.",
        flush=True,
    )
    numbers, maximum, visited = exact_map(
        table, maxima, weights, seeds, MAP_SECONDS
    )

    require(
        len(numbers) == K
        and len(set(numbers)) == K
        and all(1 <= number <= N for number in numbers),
        "Nelegalni NEXT.",
    )

    selected = np.asarray([numbers], dtype=np.int64)
    recalculated = mass_for_rows(selected, table, weights)[0]
    np.testing.assert_allclose(
        maximum, recalculated, atol=1e-14, rtol=2e-9
    )

    # Direktna bibliotečka provera baš izabrane kombinacije.
    check_library(
        final_model, x, y, next_context, selected, hx, hy
    )

    probability = maximum/z
    require(
        np.isfinite(probability) and 0 < probability <= 1+1e-9,
        "Neispravna normalizovana masa NEXT-a.",
    )

    sensitivity = 0.0
    for context in draws[-5:-1]:
        other_weights = context_weights(context[None, :], x, hx)[0]
        _, other_marginal = marginals_from_tables(
            table, sums, other_weights
        )
        sensitivity = max(
            sensitivity, float(np.max(np.abs(other_marginal-marginal)))
        )

    raw_legal_mass = z*(1-hy)**K
    require(
        np.isfinite(raw_legal_mass) and 0 < raw_legal_mass <= 1+1e-8,
        "Neispravna masa legalnog prostora u izvornoj KDE raspodeli.",
    )
    print(
        f"Masa legalnog prostora pre uslovljavanja={raw_legal_mass:.10g}; "
        f"zbir marginala={marginal.sum():.10f}; "
        f"MAP čvorovi={visited:,}.",
        flush=True,
    )

    messages = diagnose(
        numbers, draws, weights, sums, z, marginal, sensitivity
    )
    print(
        f"\nNEXT [{Path(path).name}]: "
        +" ".join(f"{number:02d}" for number in numbers),
        flush=True,
    )
    print(
        f"Interna modelska masa={probability:.12g}; "
        "nije potvrđena verovatnoća budućeg izvlačenja.",
        flush=True,
    )
    if messages:
        print(
            "STATUS: maksimum je pronađen; "
            "dijagnostika označava rezultat za proveru.",
            flush=True,
        )
    else:
        print(
            "STATUS: maksimum je pronađen; "
            "to samo po sebi ne potvrđuje prediktivnu vrednost.",
            flush=True,
        )

    return {
        "file": Path(path).name,
        "numbers": numbers,
        "marginal": marginal,
        "input_hash": fingerprint(draws),
        "hx": hx,
        "hy": hy,
        "test_nll": float(test_nll.mean()),
    }


def main():
    print(
        "Statsmodels Conditional KDE — Loto 7/39 — V1\n"
        f"Statsmodels verzija: {statsmodels.__version__}\n"
        "Diskretni Wang–Ryzin kernel; prethodno kolo -> naredno kolo.\n"
        "Cela istorija u završnom modelu; hronološka validacija.\n"
        "Bez slučajnih kombinacija i kozmetičkih filtera.",
        flush=True,
    )

    for path in PATHS:
        require(Path(path).is_file(), f"Nedostaje CSV: {path}")

    self_checks()
    results = []
    failures = []

    for path in PATHS:
        try:
            results.append(process_file(path))
        except Exception as exc:
            name = Path(path).name
            failures.append(name)
            print(
                f"\nNEUSPEH [{name}]: {type(exc).__name__}: {exc}",
                file=sys.stderr, flush=True,
            )

    for i, left in enumerate(results):
        for right in results[i+1:]:
            if left["numbers"] != right["numbers"]:
                continue
            difference = float(np.max(np.abs(
                left["marginal"]-right["marginal"]
            )))
            print(
                "\nISTI NEXT — POTREBNA PROVERA:\n"
                f"{left['file']} / {right['file']}\n"
                f"Ulazi: {left['input_hash']} / {right['input_hash']}\n"
                f"Parametri: ({left['hx']}, {left['hy']}) / "
                f"({right['hx']}, {right['hy']})\n"
                f"Maksimalna razlika marginala={difference:.9g}\n"
                "Kombinacije nisu menjane da bi bile različite.",
                flush=True,
            )

    if failures:
        print(
            "\nNije potvrđen rezultat za: "+", ".join(failures),
            flush=True,
        )
        return 1

    print("\nSva tri CSV-a obrađena.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(
            "\nPrekinuto. Nedovršena MAP pretraga ne potvrđuje maksimum.",
            file=sys.stderr,
        )
        sys.exit(130)



"""
Statsmodels Conditional KDE — Loto 7/39 — V1
Statsmodels verzija: 0.14.6
Diskretni Wang–Ryzin kernel; prethodno kolo -> naredno kolo.
Cela istorija u završnom modelu; hronološka validacija.
Bez slučajnih kombinacija i kozmetičkih filtera.
Provere kernela, statsmodels PDF-a, normalizacije i MAP-a...
Ugrađene provere prošle.


========================================================================
loto7_4706_k84.csv
Kola=4706; prelazi=4705; train=3293; validation=706; test=706.
VALIDACIJA hx=0.35, hy=0.20, NLL=29.73924638
VALIDACIJA hx=0.55, hy=0.20, NLL=26.06474906
VALIDACIJA hx=0.75, hy=0.20, NLL=23.03326801
VALIDACIJA hx=0.90, hy=0.20, NLL=21.19882292
VALIDACIJA hx=0.97, hy=0.20, NLL=20.57385027
VALIDACIJA hx=0.35, hy=0.40, NLL=23.94517776
VALIDACIJA hx=0.55, hy=0.40, NLL=21.45600610
VALIDACIJA hx=0.75, hy=0.40, NLL=19.43542824
VALIDACIJA hx=0.90, hy=0.40, NLL=18.27343176
VALIDACIJA hx=0.97, hy=0.40, NLL=17.88441229
VALIDACIJA hx=0.35, hy=0.60, NLL=20.48468648
VALIDACIJA hx=0.55, hy=0.60, NLL=18.99982206
VALIDACIJA hx=0.75, hy=0.60, NLL=17.75272503
VALIDACIJA hx=0.90, hy=0.60, NLL=17.11747148
VALIDACIJA hx=0.97, hy=0.60, NLL=16.93438212
VALIDACIJA hx=0.35, hy=0.80, NLL=18.06511162
VALIDACIJA hx=0.55, hy=0.80, NLL=17.47564761
VALIDACIJA hx=0.75, hy=0.80, NLL=16.95449888
VALIDACIJA hx=0.90, hy=0.80, NLL=16.72607981
VALIDACIJA hx=0.97, hy=0.80, NLL=16.68346011
VALIDACIJA hx=0.35, hy=0.95, NLL=16.83238382
VALIDACIJA hx=0.55, hy=0.95, NLL=16.72411447
VALIDACIJA hx=0.75, hy=0.95, NLL=16.63189010
VALIDACIJA hx=0.90, hy=0.95, NLL=16.60095028
VALIDACIJA hx=0.97, hy=0.95, NLL=16.59803538
IZBOR: hx=0.97, hy=0.95, validation NLL=16.59803538
NAPOMENA: izabran je rub pretraženog raspona parametara.
ODVOJENI TEST: prosečni NLL=16.59268891
ZAVRŠNI MODEL: svih 4705 prelaza.
MAP pretraga svih legalnih kombinacija; limit=120s.
  MAP: čvorovi=85,760, gornja/donja granica=46.6776, vreme=5.0s
Masa legalnog prostora pre uslovljavanja=3.376853434e-05; zbir marginala=7.0000000000; MAP čvorovi=168,010.
Efektivni broj istorijskih komponenti=1527.67; najveća težina=0.006957; osetljivost marginala=0.0080798993
DIJAGNOSTIKA: Kombinacija postoji 1 puta u istoriji. KDE može imati maksimum na opaženoj kombinaciji.

NEXT [loto7_4706_k84.csv]: 02 08 12 21 22 34 35
Interna modelska masa=2.07953660156e-07; nije potvrđena verovatnoća budućeg izvlačenja.
STATUS: maksimum je pronađen; dijagnostika označava rezultat za proveru.



========================================================================
loto7_4706_k84_loto_2975.csv
Kola=2975; prelazi=2974; train=2081; validation=446; test=447.
VALIDACIJA hx=0.35, hy=0.20, NLL=30.51203828
VALIDACIJA hx=0.55, hy=0.20, NLL=26.74925306
VALIDACIJA hx=0.75, hy=0.20, NLL=23.71850616
VALIDACIJA hx=0.90, hy=0.20, NLL=21.89983097
VALIDACIJA hx=0.97, hy=0.20, NLL=21.28393220
VALIDACIJA hx=0.35, hy=0.40, NLL=24.36943953
VALIDACIJA hx=0.55, hy=0.40, NLL=21.84881641
VALIDACIJA hx=0.75, hy=0.40, NLL=19.77410179
VALIDACIJA hx=0.90, hy=0.40, NLL=18.59708214
VALIDACIJA hx=0.97, hy=0.40, NLL=18.21038766
VALIDACIJA hx=0.35, hy=0.60, NLL=20.72041471
VALIDACIJA hx=0.55, hy=0.60, NLL=19.19067911
VALIDACIJA hx=0.75, hy=0.60, NLL=17.91589634
VALIDACIJA hx=0.90, hy=0.60, NLL=17.25088200
VALIDACIJA hx=0.97, hy=0.60, NLL=17.05993321
VALIDACIJA hx=0.35, hy=0.80, NLL=18.26898548
VALIDACIJA hx=0.55, hy=0.80, NLL=17.57400073
VALIDACIJA hx=0.75, hy=0.80, NLL=17.00369193
VALIDACIJA hx=0.90, hy=0.80, NLL=16.75576737
VALIDACIJA hx=0.97, hy=0.80, NLL=16.70474530
VALIDACIJA hx=0.35, hy=0.95, NLL=16.91997103
VALIDACIJA hx=0.55, hy=0.95, NLL=16.77844056
VALIDACIJA hx=0.75, hy=0.95, NLL=16.64715572
VALIDACIJA hx=0.90, hy=0.95, NLL=16.59671674
VALIDACIJA hx=0.97, hy=0.95, NLL=16.59133466
IZBOR: hx=0.97, hy=0.95, validation NLL=16.59133466
NAPOMENA: izabran je rub pretraženog raspona parametara.
ODVOJENI TEST: prosečni NLL=16.60402818
ZAVRŠNI MODEL: svih 2974 prelaza.
MAP pretraga svih legalnih kombinacija; limit=120s.
  MAP: čvorovi=137,728, gornja/donja granica=21.5632, vreme=5.0s
Masa legalnog prostora pre uslovljavanja=3.350667029e-05; zbir marginala=7.0000000000; MAP čvorovi=148,558.
Efektivni broj istorijskih komponenti=1267.79; najveća težina=0.007197; osetljivost marginala=0.01941549
DIJAGNOSTIKA: Kombinacija postoji 1 puta u istoriji. KDE može imati maksimum na opaženoj kombinaciji.

NEXT [loto7_4706_k84_loto_2975.csv]: 07 08 13 22 31 37 38
Interna modelska masa=2.25521297318e-07; nije potvrđena verovatnoća budućeg izvlačenja.
STATUS: maksimum je pronađen; dijagnostika označava rezultat za proveru.



========================================================================
loto7_4706_k84_loto_plus_1731.csv
Kola=1731; prelazi=1730; train=1211; validation=259; test=260.
VALIDACIJA hx=0.35, hy=0.20, NLL=31.46248787
VALIDACIJA hx=0.55, hy=0.20, NLL=27.83599909
VALIDACIJA hx=0.75, hy=0.20, NLL=24.77385837
VALIDACIJA hx=0.90, hy=0.20, NLL=22.87092791
VALIDACIJA hx=0.97, hy=0.20, NLL=22.20327611
VALIDACIJA hx=0.35, hy=0.40, NLL=24.65744805
VALIDACIJA hx=0.55, hy=0.40, NLL=22.35939769
VALIDACIJA hx=0.75, hy=0.40, NLL=20.31330925
VALIDACIJA hx=0.90, hy=0.40, NLL=19.07162358
VALIDACIJA hx=0.97, hy=0.40, NLL=18.63341626
VALIDACIJA hx=0.35, hy=0.60, NLL=20.65425400
VALIDACIJA hx=0.55, hy=0.60, NLL=19.32714708
VALIDACIJA hx=0.75, hy=0.60, NLL=18.13220611
VALIDACIJA hx=0.90, hy=0.60, NLL=17.43253022
VALIDACIJA hx=0.97, hy=0.60, NLL=17.20675713
VALIDACIJA hx=0.35, hy=0.80, NLL=18.11645671
VALIDACIJA hx=0.55, hy=0.80, NLL=17.49695528
VALIDACIJA hx=0.75, hy=0.80, NLL=17.00833592
VALIDACIJA hx=0.90, hy=0.80, NLL=16.76824385
VALIDACIJA hx=0.97, hy=0.80, NLL=16.71210410
VALIDACIJA hx=0.35, hy=0.95, NLL=16.85698025
VALIDACIJA hx=0.55, hy=0.95, NLL=16.73973452
VALIDACIJA hx=0.75, hy=0.95, NLL=16.62623665
VALIDACIJA hx=0.90, hy=0.95, NLL=16.58855628
VALIDACIJA hx=0.97, hy=0.95, NLL=16.58755703
IZBOR: hx=0.97, hy=0.95, validation NLL=16.58755703
NAPOMENA: izabran je rub pretraženog raspona parametara.
ODVOJENI TEST: prosečni NLL=16.58062192
ZAVRŠNI MODEL: svih 1730 prelaza.
MAP pretraga svih legalnih kombinacija; limit=120s.
Masa legalnog prostora pre uslovljavanja=3.308698133e-05; zbir marginala=7.0000000000; MAP čvorovi=68,361.
Efektivni broj istorijskih komponenti=712.86; najveća težina=0.012236; osetljivost marginala=0.021192785
DIJAGNOSTIKA: Kombinacija postoji 1 puta u istoriji. KDE može imati maksimum na opaženoj kombinaciji.

NEXT [loto7_4706_k84_loto_plus_1731.csv]: 05 09 10 22 24 30 31
Interna modelska masa=3.20287113344e-07; nije potvrđena verovatnoća budućeg izvlačenja.
STATUS: maksimum je pronađen; dijagnostika označava rezultat za proveru.

Sva tri CSV-a obrađena.
"""



"""
KDEMultivariateConditional. Statsmodels. 

Uslovna procena raspodele — KDEMultivariateConditional
- Uči zajedničku raspodelu izlaza uslovljenu ulazom, pa prethodna kola mogu biti ulaz, a naredno kolo izlaz.
- Podržava diskretne promenljive.
- Ne zahteva pretpostavku da je ciljna raspodela normalna i nije klasifikator. 
Ograničenje: sama biblioteka ne obezbeđuje sedam različitih brojeva od 1 do 39. Potrebno je proveriti uslovljavanje na legalne kombinacije, normalizaciju i cenu traženja maksimuma. Ugrađena unakrsna validacija takođe ne sme automatski zameniti hronološku validaciju.

V1 sa izvornom uslovnom KDE metodom i diskretnim kernelima. Izbor parametara je hronološki, završno učenje koristi celu istoriju, a NEXT se traži među legalnim kombinacijama — bez zaokruživanja i popravljanja duplikata.

V1 zadržava uslovni KDE i originalni Wang-Ryzin kernel za diskretne brojeve. 
Moja adaptacija je normalizacija preko legalnih kombinacija 7/39 i hronološki izbor parametara. 
Ugrađene provere porede račun sa instaliranim statsmodels i potpunim nabrajanjem malog primera. 
MAP pretraga ima limit od 120 sekundi po CSV-u; ako ne potvrdi maksimum, prijavljuje neuspeh umesto izmišljene NEXT kombinacije.
"""
