import inspect

def call_generate_signal(
    obj, *,
    df=None,
    current_price=None,
    context=None,
    indicators=None,
    current_bar=None,
    **_ignore,  # ignore any future extra kwargs to avoid TypeErrors
):
    """Call obj.generate_signal with only the parameters it declares.
    Supports strategies that want a 'current_bar' (or alias) argument.
    """
    fn = getattr(obj, "generate_signal", None)
    if fn is None:
        raise AttributeError(f"{obj} has no generate_signal()")

    allowed = inspect.signature(fn).parameters
    kwargs = {}
    if "df" in allowed: kwargs["df"] = df
    if "current_price" in allowed: kwargs["current_price"] = current_price
    if "context" in allowed: kwargs["context"] = context
    if "indicators" in allowed: kwargs["indicators"] = indicators

    # ORB / bar-driven strategies
    for alias in ("current_bar", "bar", "bar_data", "latest_bar"):
        if alias in allowed:
            kwargs[alias] = current_bar
            break

    return fn(**kwargs)
