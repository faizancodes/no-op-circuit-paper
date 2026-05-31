        time_strings = TIME_STRINGS
    if depth <= 0:
        raise ValueError("depth must be greater than 0.")
    # Convert datetime.date to datetime.datetime for comparison.
    if not isinstance(d, datetime.datetime):
        d = datetime.datetime(d.year, d.month, d.day)
    if now and not isinstance(now, datetime.datetime):
        now = datetime.datetime(now.year, now.month, now.day)

    now = now or datetime.datetime.now(datetime.timezone.utc if is_aware(d) else None)

    if reversed:
        d, now = now, d
    delta = now - d

    # Ignore microseconds.
    since = delta.days * 24 * 60 * 60 + delta.seconds
    if since <= 0:
        # d is in the future compared to now, stop processing.
        return avoid_wrapping(time_strings["minute"] % {"num": 0})

    # Get years and months.
    total_months = (now.year - d.year) * 12 + (now.month - d.month)
    if d.day > now.day or (d.day == now.day and d.time() > now.time()):
        total_months -= 1
    years, months = divmod(total_months, 12)

    # Calculate the remaining time.
    # Create a "pivot" datetime shifted from d by years and months, then use
    # that to determine the other parts.
    if years or months:
        pivot_year = d.year + years
        pivot_month = d.month + months
        if pivot_month > 12:
            pivot_month -= 12
            pivot_year += 1
        pivot = datetime.datetime(
            pivot_year,
            pivot_month,
            min(MONTHS_DAYS[pivot_month - 1], d.day),
            d.hour,
            d.minute,
            d.second,
        )
    else:
        pivot = d
    remaining_time = (now - pivot).total_seconds()
    partials = [years, months]
    for chunk in TIME_CHUNKS:
        count = remaining_time // chunk
        partials.append(count)
        remaining_time -= chunk * count

    # Find the first non-zero part (if any) and then build the result, until
    # depth.
    i = 0
    for i, value in enumerate(partials):
        if value != 0:
            break
    else:
        return avoid_wrapping(time_strings["minute"] % {"num": 0})

    result = []
    current_depth = 0
    while i < len(TIME_STRINGS_KEYS) and current_depth < depth:
        value = partials[i]
        if value == 0:
            break
        name = TIME_STRINGS_KEYS[i]
        result.append(avoid_wrapping(time_strings[name] % {"num": value}))
        current_depth += 1
        i += 1

    return gettext(", ").join(result)


def timeuntil(d, now=None, time_strings=None, depth=2):
    """
    Like timesince, but return a string measuring the time until the given time.
    """
    return timesince(d, now, reversed=True, time_strings=time_strings, depth=depth)
