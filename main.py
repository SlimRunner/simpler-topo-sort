from queue import PriorityQueue


def main():
    graph = [
        ("x4", "x1"),
        ("x5", "x4"),
        ("z1", "x5"),
        ("z2", "z1"),
        ("z3", "z2"),
        ("x3", "x2"),
        ("x6", "w2"),
        ("x7", "w2"),
        ("x1", "x2"),
        ("x2", "x1"),
        ("a1", "a2"),
        ("a2", "a3"),
        ("a3", "a4"),
        ("a4", "a1"),
    ]
    # graph = [
    #     ("a2", "a2"),
    #     ("a3", "t0"),
    #     ("a4", "a7"),
    #     ("a5", "a6"),
    #     ("a6", "a5"),
    #     ("a7", "a4"),
    # ]
    graph = [
        ("a2", "t3"),
        ("a3", "a7"),
        ("a4", "a3"),
        ("a5", "a4"),
        ("a6", "a5"),
        ("a7", "a5")
    ]

    for lhs, rhs in createScheduleWithCycles(graph):
        print(f"{lhs} = {rhs}")


def createScheduleWithCycles(tasks: list[tuple[str, str]]):
    # similar to createSchedule but can handle cycels of rank 1
    subSchedules = []
    schedule = []
    for comp in separateComps(tasks):
        subSchedules.append(createSchedule(comp))
    for task in subSchedules:
        for lhs, rhs in reversed(task):
            schedule.append((lhs, rhs))

    return schedule


def separateComps(tasks: list[tuple[str, str]]):
    graph: dict[str, str] = dict()
    parset: set[str] = set()

    queue = PriorityQueue()

    for k, v in tasks:
        if k == v:
            continue
        graph[k] = v
        parset.add(v)

        # queue.append(k)

    for k, v in graph.items():
        if k in parset:
            queue.put((1, k))
        else:
            queue.put((0, k))

    viset: set[str] = set()
    chains: list[list[tuple[str, str]]] = []
    cycles: list[list[tuple[str, str]]] = []

    while not queue.empty():
        k = queue.get()[1]
        stack: list[str] = [k]
        while k not in viset and k in graph:
            viset.add(stack[-1])
            k = graph[k]
            stack.append(k)
        halt_at = k
        if len(stack) >= 2:
            isCycle = False
            tempout = []

            while len(stack) >= 2:
                tempout.append((stack[-2], stack[-1]))
                stack.pop()
                k = stack[-1]
                if halt_at == k:
                    isCycle = True
                    stack.clear()
                    break
            if isCycle:
                cycles.append(tempout)
            else:
                chains.append(tempout)

    return chains + cycles


def createSchedule(tasklist: list[tuple[str, str]]):
    if len(tasklist) == 0:
        return

    # map of renames: new -> old
    graph: dict[str, str] = {}
    visited: set[str] = set()

    # NOTE: this function assumes that neither the set of old names or
    # the set of new names contain any duplicates. The implication is
    # that the resulting graph has no branching.

    # build graph
    for fsrc, ftrg in tasklist:
        if fsrc == ftrg:
            continue
        # detect branching
        if ftrg in graph:
            raise ValueError("Pattern provided does not yield unique names.")
        graph[ftrg] = fsrc

    cycles: list[str] = []
    sequences: list[str] = []
    seqdict: set[str] = set()

    # categorize connected components
    for node in graph:
        if node not in visited:
            if node == graph[node]:
                # ignore loops: meaning name does not change
                continue
            # repNode is the node which completes a cycle (None otherwise)
            # seqNode is the node that breaks a sequence (visited)
            repNode, seqNode = identifyCycle(graph, visited, node)
            if repNode is not None:
                cycles.append(node)
            else:
                if seqNode in seqdict:
                    seqdict.remove(seqNode)
                seqdict.add(node)

    sequences = list(seqdict)

    schedule: list[tuple[str, str]] = []
    tempName: str | None = None

    # process acyclic chains
    for seq in sequences:
        node = seq
        while node in graph:
            schedule.append((graph[node], node))
            node = graph[node]
        tempName = node

    if tempName is None and len(sequences) == 0 and len(cycles) > 0:
        tempName = "__tmp__"

    # process cyclic chains
    for seed in cycles:
        node = graph[seed]
        schedule.append((seed, tempName))
        schedule.append((node, seed))
        while node != seed:
            schedule.append((graph[node], node))
            node = graph[node]
        _, tail = schedule[-1]
        schedule[-1] = (tempName, tail)

    return schedule


def identifyCycle(
    graph: dict[str, str], visited: set[str], seed: str
) -> tuple[str | None, str | None]:
    # this function can only handle graphs without branching
    visited.add(seed)
    prev = None
    node = graph[seed]
    while node is not None and node != seed:
        if node not in graph or node in visited:
            visited.add(node)
            prev = node
            node = None
        else:
            visited.add(node)
            node = graph[node]

    return (node, prev)


if __name__ == "__main__":
    main()
