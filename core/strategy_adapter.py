import inspect

def call_generate_signal(obj, *, df=None, current_price=None, context=None, indicators=None):
    """Call obj.generate_signal with only the params it declares."""
    fn = getattr(obj, "generate_signal", None)
    if fn is None:
        raise AttributeError(f"{obj} has no generate_signal()")  # pragma: no cover

    allowed = inspect.signature(fn).parameters
    kwargs = {}
    if "df" in allowed: kwargs["df"] = df
    if "current_price" in allowed: kwargs["current_price"] = current_price
    if "context" in allowed: kwargs["context"] = context
    if "indicators" in allowed: kwargs["indicators"] = indicators
    # NEW: pass the latest bar if the strategy wants it
    for alias in ("current_bar", "bar", "bar_data", "latest_bar"):
        if alias in allowed:
            kwargs[alias] = current_bar
            break
    return fn(**kwargs)
