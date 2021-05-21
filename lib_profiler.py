import cProfile, pstats, io


def profile(fn):
    """Uses cProfile to profile a function
    To use:
    from profiler import profile
    add @profile decorator to the function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fn(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()

        with open("output/results/{ID}_profile_result.txt", "w+") as f:
            f.write(s.getvalue())

        return retval

    return inner
