def root_print(comm, *args, **kwargs) -> None:
    if comm.rank == 0:
        print(*args, **kwargs, flush=True)