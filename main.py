import sys
from pprint import pformat
from collections import Counter
from collections.abc import Callable


def main():
    runTests()


def createSchedule(tasks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    repls = [(rhs, lhs) for lhs, rhs in tasks]
    graph, revGraph, parents, context, outDeg, inDeg = makeGraph(
        repls, filter=lambda k, v: k != v
    )

    if len(graph) == 0:
        return list()

    if inDeg != 1:
        raise ValueError("Input graph must have max in-degree of exactly 1")

    visits: Counter[str] = Counter()
    orderedChains: list[tuple[list[str], bool]] = []

    for node, _ in reversed(parents.most_common()):
        if visits[node] != 0:
            continue

        # this call relies on revGraph having max out-degree of exactly 1
        orderedChains.extend(findComponents(revGraph, node, visits))

    schedule = topoSortChains(orderedChains, context)

    return schedule


def topoSortChains(chains: list[tuple[list[str], bool]], context: set[str]):
    schedule: list[tuple[str, str]] = []
    if len(chains) == 0:
        return schedule

    tasks: list[list[str]] = [[] for _ in chains]

    for ith, root in enumerate(chains):
        if len(root):
            stack = [(ith, 0)]
        while stack:
            i, j = stack.pop()
            chIdx = j
            task = tasks[i]
            chain, isCycle = chains[i]
            restKeys = chain[j:]

            # add cycle breaker start
            if isCycle and len(task) == 0:
                task.append(genTempName(context))

            isDone = True
            for key in restKeys:
                # check dependencies
                nextDep = depLookAhead(chains, key, i)

                # if so fulfill them first
                if nextDep is not None:
                    stack.append((i, chIdx))
                    stack.append((nextDep, 0))
                    isDone = False
                    break

                # otherwise aggregate node
                task.append(key)
                chIdx += 1

            # add cycle breaker end
            if isCycle and isDone:
                task.append(task[0])

            # dump task list into final schedule
            if isDone:
                chain.clear()
                while len(task) > 1:
                    lhs = task.pop()
                    rhs = task[-1]
                    schedule.append((lhs, rhs))
                task.clear()

    return schedule


def depLookAhead(chains: list[tuple[list[str], bool]], key: str, at: int):
    idx = at + 1
    for chain, _ in chains[at + 1 :]:
        if len(chain) > 0 and chain[0] == key:
            return idx
        idx += 1
    return None


def findComponents(revGraph: dict[str, list[str]], root: str, visits: Counter[str]):
    ordered: list[tuple[list[str], bool]] = []
    stack: list[str] = [root]
    visits[root] += 1

    while stack:
        node = stack[-1]

        if node not in revGraph or visits[node] > 1:
            # end of chain encountered
            ordered.append(sortPath(stack, visits))
            continue

        nextNode = revGraph[node][0]

        if visits[nextNode] == 1:
            # cycle encountered
            ordered.append(sortCycle(stack, visits, nextNode))
            continue

        stack.append(nextNode)
        visits[nextNode] += 1

    return ordered


def sortCycle(stack: list[str], visits: Counter[str], repNode: str):
    chain: list[str] = []

    while stack:
        node = stack[-1]
        visits[node] += 1
        if node != repNode or len(stack) == 1:
            chain.append(stack.pop())
        else:
            chain.append(node)
            break

    return chain, True


def sortPath(stack: list[str], visits: Counter[str]):
    chain: list[str] = []

    while stack:
        node = stack.pop()
        visits[node] += 1
        chain.append(node)

    return chain, False


def makeGraph(
    edges: list[tuple[str, str]], filter: Callable[[str, str], bool] | None = None
):
    graph: dict[str, list[str]] = dict()
    revGraph: dict[str, list[str]] = dict()
    parents: Counter[str] = Counter()
    context: set[str] = set()
    maxOutDegree = 0
    maxInDegree = 0

    srcNodes: Counter[str] = Counter()
    trgNodes: Counter[str] = Counter()

    for parent, child in edges:
        context.add(parent)
        context.add(child)
        if filter is not None and not filter(parent, child):
            continue

        srcNodes[parent] += 1
        trgNodes[child] += 1
        parents[parent] += 1
        parents[child] += 0

        maxOutDegree = max(maxOutDegree, srcNodes[parent])
        maxInDegree = max(maxInDegree, trgNodes[child])

        if parent in graph:
            graph[parent].append(child)
        else:
            graph[parent] = [child]

        if child in revGraph:
            revGraph[child].append(parent)
        else:
            revGraph[child] = [parent]

    return graph, revGraph, parents, context, maxOutDegree, maxInDegree


def genTempName(context: set[str]):
    result = "__temp__"
    if result in context:
        raise ValueError(f"{result} is not unique in context")

    return result


def runTests():
    msglines = []

    for taskList in TESTS:
        values: dict[str, int] = dict()
        expected: dict[str, int] = dict()
        saved: dict[str, int] = dict()
        uniqval = 0

        for lhs, rhs in taskList:
            if lhs not in values:
                values[lhs] = uniqval
                uniqval += 1
            if rhs not in values:
                values[rhs] = uniqval
                uniqval += 1
            saved[lhs] = values[lhs]

        for lhs, rhs in taskList:
            if rhs not in saved:
                expected[lhs] = values[rhs]
            else:
                expected[lhs] = saved[rhs]

        msgerr = ""
        msgerr += pformat(taskList) + "\n"
        schedule = createSchedule(taskList)
        msgerr += pformat([f"{l} = {r}" for l, r in schedule]) + "\n"
        msglines.append(pformat([f"{l} = {r}" for l, r in schedule]))

        for lhs, rhs in schedule:
            values[lhs] = values[rhs]

        for k, v in expected.items():
            msgerr += f"key, value: {k}, {v}\n"
            assertEquals(values[k], v, msgerr)

    print("\n\n".join(msglines))


def assertEquals(value, expected, msg: str | None = None):
    if value != expected:
        if msg:
            print(msg, file=sys.stderr)
        raise AssertionError(f"value {value} does not equal {expected}")


TESTS: list[list[tuple[str, str]]] = [
    [
        ("x2", "x1"),
        ("x1", "x0"),
        ("x0", "y0"),
        ("y0", "z0"),
        ("z0", "x0"),
        ("y1", "y0"),
        ("y2", "y1"),
        ("y3", "y1"),
        ("y4", "y3"),
        ("y5", "y4"),
        ("y6", "y5"),
        ("y7", "y6"),
        ("y8", "y6"),
        ("y9", "y6"),
        ("w0", "w0"),
    ],
    [
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
    ],
    [
        ("a2", "a2"),
        ("a3", "t0"),
        ("a4", "a7"),
        ("a5", "a6"),
        ("a6", "a5"),
        ("a7", "a4"),
    ],
    [
        ("a2", "t3"),
        ("a3", "a7"),
        ("a4", "a3"),
        ("a5", "a4"),
        ("a6", "a5"),
        ("a7", "a5"),
    ],
    [
        ("a2", "t3"),
        ("a3", "a7"),
        ("a4", "a3"),
        ("a5", "a4"),
        ("a6", "a5"),
        ("a7", "a5"),
    ],
    [
        ("v0", "v1"),
        ("v1", "v2"),
        ("v2", "v3"),
        ("v3", "v4"),
        ("v4", "v0"),
    ],
    [("x", "x")],
    [
        ("x", "x"),
        ("x", "y"),
    ],
    [],
]

if __name__ == "__main__":
    main()
