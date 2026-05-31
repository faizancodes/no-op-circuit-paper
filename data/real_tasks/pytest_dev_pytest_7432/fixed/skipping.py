        item._store[xfailed_key] = xfailed = evaluate_xfail_marks(item)

    if not item.config.option.runxfail:
        if xfailed and not xfailed.run:
            xfail("[NOTRUN] " + xfailed.reason)

    yield


@hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Item, call: CallInfo[None]):
    outcome = yield
    rep = outcome.get_result()
    xfailed = item._store.get(xfailed_key, None)
    # unittest special case, see setting of unexpectedsuccess_key
    if unexpectedsuccess_key in item._store and rep.when == "call":
        reason = item._store[unexpectedsuccess_key]
        if reason:
            rep.longrepr = "Unexpected success: {}".format(reason)
        else:
            rep.longrepr = "Unexpected success"
        rep.outcome = "failed"
    elif item.config.option.runxfail:
        pass  # don't interfere
    elif call.excinfo and isinstance(call.excinfo.value, xfail.Exception):
        assert call.excinfo.value.msg is not None
        rep.wasxfail = "reason: " + call.excinfo.value.msg
        rep.outcome = "skipped"
    elif not rep.skipped and xfailed:
        if call.excinfo:
            raises = xfailed.raises
            if raises is not None and not isinstance(call.excinfo.value, raises):
                rep.outcome = "failed"
            else:
                rep.outcome = "skipped"
                rep.wasxfail = xfailed.reason
        elif call.when == "call":
            if xfailed.strict:
                rep.outcome = "failed"
                rep.longrepr = "[XPASS(strict)] " + xfailed.reason
            else:
                rep.outcome = "passed"
                rep.wasxfail = xfailed.reason

    if (
        item._store.get(skipped_by_mark_key, True)
        and rep.skipped
        and type(rep.longrepr) is tuple
    ):
        # skipped by mark.skipif; change the location of the failure
        # to point to the item definition, otherwise it will display
        # the location of where the skip exception was raised within pytest
        _, _, reason = rep.longrepr
        filename, line = item.reportinfo()[:2]
        assert line is not None
        rep.longrepr = str(filename), line + 1, reason


def pytest_report_teststatus(report: BaseReport) -> Optional[Tuple[str, str, str]]:
    if hasattr(report, "wasxfail"):
        if report.skipped:
            return "xfailed", "x", "XFAIL"
        elif report.passed:
            return "xpassed", "X", "XPASS"
    return None
