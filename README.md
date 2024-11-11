# UCS2504 - Foundations of Artificial Intelligence
## Assignment 1
**Date :** 01-08-2024

**Problem Description :**

1 Representing Search Problems :

A search problem consists of
- a start node
- a neighbors function that, given a node, returns an enumeration of the edges from the
node
- a specification of a goal in terms of a Boolean function that takes a node and returns true
if the node is a goal
- a (optional) heuristic function that, given a node, returns a non-negative real number. The
heuristic function defaults to zero.

As far as the searcher is concerned a node can be anything. In the simple examples, the node is a string. Define an abstract class Search problem with methods start node(), is goal(),
neighbors() and heuristic().

The neighbors is a list of edges. A (directed) edge consists of two nodes, a from node and a
to node. The edge is the pair (from node,to node), but can also contain a non-negative cost
(which defaults to 1) and can be labeled with an action. Implement a class Edge. Define a suitable repr () method to print the edge.

2 Explicit Representation of Search Graph :

The first representation of a search problem is from an explicit graph (as opposed to one that is
generated as needed). An explicit graph consists of
- a set of nodes
- a list of edges
- a start node
- a set of goal nodes
- (optionally) a dictionary that maps a node to a heuristic value for that node

To define a search problem, we need to define the start node, the goal predicate, the neighbors function and the heuristic function. Define a concrete class
Search problem from explicit graph(Search problem).

Give a title string also to the search problem. Define a suitable repr () method to print the
graph.

3 Paths :

A searcher will return a path from the start node to a goal node. Represent the path in terms of a recursive data structure that can share subparts. A path is either:
- a node (representing a path of length 0) or
- an initial path and an edge, where the from node of the edge is the node at the end of
initial.

Implement a class Path(). Define a suitable repr () method to print the path.

4 Example Search Problems :

Using Search problem from explicit graph, represent the following graphs.

For example, the first graph can be created with the code
from searchProblem import Edge, Search_problem_from_explicit_graph,
Search_problem
problem1 = Search_problem_from_explicit_graph(’Problem 1’,
{’A’,’B’,’C’,’D’,’G’},
[Edge(’A’,’B’,3), Edge(’A’,’C’,1), Edge(’B’,’D’,1), Edge(’B’,’G’,3),
Edge(’C’,’B’,1), Edge(’C’,’D’,3), Edge(’D’,’G’,1)],
start = ’A’,
goals = {’G’})

5 Searcher :

A Searcher for a problem is given can be asked repeatedly for the next path. To solve a problem, you can construct a Searcher object for the problem and then repeatedly ask for the next path using search. If there are no more paths, None is returned. Implement Searcher class with DFS (Depth-First Search).

To use depth-first search to find multiple paths for problem1, copy and paste the following
into Python’s read-evaluate-print loop; keep finding next solutions until there are no more:

Depth-first search for problem1; do the following:
searcher1 = Searcher(searchExample.problem1)
searcher1.search() # find first solution
searcher1.search() # find next solution (repeat until no solutions)

**Algorithm:**
```bash
Input: problem
Output: solution, or failure

frontier ← [initial state of problem]
explored = {}
while frontier is not empty do
  node ← remove a node from frontier
  if node is a goal state then
    return solution
  end
  add node to explored
  add the successor nodes to frontier only if not in frontier or explored
end
return failure
```
**Code :** 
```python
class Edge:
    def __init__(self, start_node, end_node, cost):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = cost

    def __repr__(self):
        return f"{self.start_node}{self.end_node}"

class Search_problem_from_explicit_graph:
    def __init__(self, title, nodes, graph, start, goals):
        self.title = title
        self.nodes = nodes
        self.edges = graph
        self.start = start
        self.goals = goals

    def is_goal(self, in_node):
        return in_node in self.goals

    def neighbours(self, node):
        near_neighbours = []
        for e in self.edges:
            if e.start_node == node:
                near_neighbours.append(e)
        return near_neighbours

    def start_node(self):
        return self.start

class Path:
    def __init__(self, initial=None, edge=None, cost=0):
        self.initial = initial
        self.edge = edge
        self.cost = cost

    def end(self):
        if self.edge is None:
            return self.initial
        else:
            return self.edge.end_node

    def __repr__(self):
        if self.edge is None:
            return f"{self.initial} -> "
        else:
            return f"{self.initial} {self.edge} -> "

class Searcher:
    def __init__(self, problem):
        self.problem = problem
        self.yet_to_explore = [Path(problem.start)]
        self.visited = set()

    def search(self):
        while self.yet_to_explore:
            path = self.yet_to_explore.pop()
            node = path.end()
            if self.problem.is_goal(node):
                print(path)
                print("Cost of the path:", path.cost)
                print("\n")
                return
            if node not in self.visited:
                self.visited.add(node)
                for edge in self.problem.neighbours(node):
                    new_path = Path(path, edge, path.cost + edge.weight)
                    self.yet_to_explore.append(new_path)
                self.visited.remove(node)
        return None

problem1 = Search_problem_from_explicit_graph("Problem 1", {"A", "B", "C", "D", "G"},
[Edge("A", "B", 3), Edge("A", "C", 1), Edge("B", "D", 1), Edge("B", "G", 3),
Edge("C", "B", 1), Edge("C", "D", 3), Edge("D", "G", 1)],
start="A",
goals={"G", "B"})

searcher1 = Searcher(problem1)
print("1All possible paths from source to goal node:")
searcher1.search()
searcher1.search()

print("2All possible paths from source to goal node:")
problem2 = Search_problem_from_explicit_graph("Problem 2", {"A", "B", "C", "D", "E", "G", "H", "J"},
[Edge("A", "B", 1), Edge("A", "H", 3), Edge("B", "C", 3), Edge("B", "D", 1), Edge("D", "E", 3),
Edge("H", "J", 1), Edge("D", "G", 1)],
start="A",
goals={"G", "H"})

searcher2 = Searcher(problem2)
searcher2.search()

```
**Testing :**
```bash
runfile('C:/Users/niran/Desktop/COLLEGE/3/AI/ai.py', wdir='C:/Users/niran/Desktop/COLLEGE/3/AI')
1All possible paths from source to goal node:
A ->  AC ->  CD ->  DG -> 
Cost of the path: 5


A ->  AC ->  CB -> 
Cost of the path: 2


2All possible paths from source to goal node:
A ->  AH -> 
Cost of the path: 3
```
<div style="page-break-after: always;"></div>

## Assignment 2
**Date :** 08-08-2024

**Problem Description :**

1 Representing Search Problems :

A search problem consists of
- a start node
- a neighbors function that, given a node, returns an enumeration of the edges from the
node
- a specification of a goal in terms of a Boolean function that takes a node and returns true
if the node is a goal
- a (optional) heuristic function that, given a node, returns a non-negative real number. The
heuristic function defaults to zero.

As far as the searcher is concerned a node can be anything. In the simple examples, the node is a string. Define an abstract class Search problem with methods start node(), is goal(),
neighbors() and heuristic().

The neighbors is a list of edges. A (directed) edge consists of two nodes, a from node and a
to node. The edge is the pair (from node,to node), but can also contain a non-negative cost
(which defaults to 1) and can be labeled with an action. Implement a class Edge. Define a suitable repr () method to print the edge.

2 Explicit Representation of Search Graph :

The first representation of a search problem is from an explicit graph (as opposed to one that is
generated as needed). An explicit graph consists of
- a set of nodes
- a list of edges
- a start node
- a set of goal nodes
- (optionally) a dictionary that maps a node to a heuristic value for that node

To define a search problem, we need to define the start node, the goal predicate, the neighbors function and the heuristic function. Define a concrete class
Search problem from explicit graph(Search problem).

Give a title string also to the search problem. Define a suitable repr () method to print the
graph.
3 Paths :

A searcher will return a path from the start node to a goal node. Represent the path in terms of a recursive data structure that can share subparts. A path is either:
- a node (representing a path of length 0) or
- an initial path and an edge, where the from node of the edge is the node at the end of
initial.

Implement a class Path(). Define a suitable repr () method to print the path.

4 Example Search Problems :

Using Search problem from explicit graph, represent the following graphs.

For example, the first graph can be created with the code
from searchProblem import Edge, Search_problem_from_explicit_graph,
Search_problem
problem1 = Search_problem_from_explicit_graph(’Problem 1’,
{’A’,’B’,’C’,’D’,’G’},
[Edge(’A’,’B’,3), Edge(’A’,’C’,1), Edge(’B’,’D’,1), Edge(’B’,’G’,3),
Edge(’C’,’B’,1), Edge(’C’,’D’,3), Edge(’D’,’G’,1)],
start = ’A’,
goals = {’G’})

5 Frontier as a Priority Queue
In many of the search algorithms, such as Uniform Cost Search, A* and other best-first searchers, the frontier is implemented as a priority queue. Use Python’s built-in priority queue implementations heapq (read the Python documentation, https://docs.python. org/3/library/heapq.html).
Implement FrontierPQ. A frontier is a list of triples. The first element of each triple is the value to be minimized. The second element is a unique index which specifies the order that the elements were added to the queue, and the third element is the path that is on the queue. The use of the unique index ensures that the priority queue implementation does not compare paths; whether one path is less than another is not defined. It also lets us control what sort of search (e.g., depth-first or breadth-first) occurs when the value to be minimized does not give a unique next path. Use a variable frontier index to maintain the total number of elements of the frontier that have been created.


6 Searcher
A Searcher for a problem can be asked repeatedly for the next path. To solve a problem, you can construct a Searcher object for the problem and then repeatedly ask for the next path using search. If there are no more paths, None is returned. Implement Searcher class using using the FrontierPQ class.


**Algorithm:**
```bash
Input: problem
Output: solution, or failure

frontier ← Priority Queue
Add starting node to frontier
explored ← Set
while frontier is not empty do
  path ← remove the frontier node with shortest distance
  v ← path.node
  if v is a goal node then return solution
  if v is not in explored
    for each successor w of v do
	    new_path ← path + v
	    new_cost ← path.cost + heuristic(u)
	    Add new_path to Frontier
return failure
```
**Code :** 
```python
import heapq

class Edge:
    def __init__(self, start_node, end_node, cost):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = cost

    def __repr__(self):
        return f"{self.start_node},{self.end_node}"

class Search_problem_from_explicit_graph:
    def __init__(self, title, nodes, graph, start, goals):
        self.title = title
        self.nodes = nodes
        self.edges = graph
        self.start = start
        self.goals = goals

    def is_goal(self, in_node):
        return in_node in self.goals

    def neighbours(self, node):
        near_neighbours = []
        for e in self.edges:
            if e.start_node == node:
                near_neighbours.append(e)
        return near_neighbours

    def start_node(self):
        return self.start

class Path:
    def __init__(self, initial=None, edge=None, cost=0):
        self.initial = initial  
        self.edge = edge
        self._cost = cost 

    def end(self):
        if self.edge is None:
            return self.initial
        else:
            return self.edge.end_node

    def cost(self):
        return self._cost

    def __repr__(self):
        if self.edge is None:
            return f"{self.initial} -> "
        else:
            return f"{self.initial} {self.edge} -> "

class FrontierPQ:
    def __init__(self):
        self.counter = 0
        self.yet_to_explore = []

    def add(self, cost, path):
        heapq.heappush(self.yet_to_explore, (cost, self.counter, path))
        self.counter += 1

    def pop(self):
        if self.yet_to_explore:
            return heapq.heappop(self.yet_to_explore)[-1]
        return None

    def print_frontier(self):
        print("Current frontier:")
        for item in self.yet_to_explore:
            cost, counter, path = item
            print(f"Cost: {cost}, Counter: {counter}, Path: {path}")

class Searcher:
    def __init__(self, problem):
        self.problem = problem
        self.frontier = FrontierPQ()
        self.frontier.add(0, Path(problem.start_node))
        self.visited = set()
        self.goals_reached = set()
        self.no_of_goals = len(problem.goals)

    def search(self):
        while self.frontier:
            path = self.frontier.pop()  

            if path is None:
                print("No more paths to explore.")
                return None

            node = path.end()

            if node not in self.goals_reached and node in self.problem.goals:
                self.goals_reached.add(node)
                print("To reach:", node)
                print(path)
                print("Cost of the path:", path.cost())
                print("\n")

            if len(self.goals_reached) == self.no_of_goals:
                return

            if node not in self.visited:
                self.visited.add(node)
                for edge in self.problem.neighbours(node):
                    new_path = Path(path, edge, path.cost() + edge.weight)
                    self.frontier.add(new_path.cost(), new_path)
                self.visited.remove(node)

        return None


problem1 = Search_problem_from_explicit_graph("Problem 1", {"A", "B", "C", "D", "G"},
    [Edge("A", "B", 3), Edge("A", "C", 1), Edge("B", "D", 1), Edge("B", "G", 3),
     Edge("C", "B", 1), Edge("C", "D", 3), Edge("D", "G", 1)],
    start="A", goals={"G"})

searcher1 = Searcher(problem1)
print("All possible paths from source to goal node:")
searcher1.search()



print("All possible paths from source to goal node:")
problem2 = Search_problem_from_explicit_graph("Problem 2", {"A", "B", "C", "D", "E", "G", "H", "J"},
    [Edge("A", "B", 1), Edge("A", "H", 3), Edge("B", "C", 3), Edge("B", "D", 1), Edge("D", "E", 3),
     Edge("H", "J", 1), Edge("D", "G", 1)],
    start="A", goals={"G", "H"})

searcher2 = Searcher(problem2)
searcher2.search()
        
```
**Testing :**
```bash
runfile('C:/Users/niran/Desktop/COLLEGE/3/AI/ai.py', wdir='C:/Users/niran/Desktop/COLLEGE/3/AI')
All possible paths from source to goal node:
To reach: G
A ->  A,C ->  C,B ->  B,D ->  D,G -> 
Cost of the path: 4


All possible paths from source to goal node:
To reach: H
A ->  A,H -> 
Cost of the path: 3


To reach: G
A ->  A,B ->  B,D ->  D,G -> 
Cost of the path: 3
```

<div style="page-break-after: always;"></div>


## Assignment 3
**Date :** 12-0-2024

**Problem Description 1:**

In a 3×3 board, 8 of the squares are filled with integers 1 to 9, and one square is left empty. One move is sliding into the empty square the integer in any one of its adjacent squares. The start state is given on the left side of the figure, and the goal state given on the right side. Find a sequence of moves to go from the start state to the goal state.

1) Formulate the problem as a state space search problem.
2) Find a suitable representation for the states and the nodes.
3) Solve the problem using any of the uninformed search strategies.
4) We can use Manhattan distance as a heuristic h(n). The cheapest cost from the current node to the goal node, can be estimated as how many moves will be required to transform the current node into the goal node. This is related to the distance each tile must travel to arrive at its destination, hence we sum the Manhattan distance of each square from its home position.
5) An alternative heuristic should consider the number of tiles that are “out-of-sequence”.
An out of sequence score can be computed as follows:
- a tile in the center counts 1,
- a tile not in the center counts 0 if it is followed by its proper successor as defined
by the goal arrangement,
- otherwise, a tile counts 2.
6) Use anyone of the two heuristics, and implement Greedy Best-First Search.
7) Use anyone of the two heuristics, and implement A* Search



**Algorithm:**
1. A* 
```bash
Input: problem
Output: solution, or failure

frontier ← Priority Queue
add starting node to frontier with priority = heuristic(start) + 0  

while frontier is not empty do
    path ← remove node from frontier with lowest priority
    node ← path.node
    add node to explored set
    
    for each neighbor of node do
        if neighbor not in explored set then
            new_path ← Path(neighbor, path, edge)
            if neighbor is a goal node then
                return new_path as solution
            
            frontier.add((heuristic(neighbor) + g(new_path), new_path))
            
return failure
```
2. Greedy Best First Search 
```bash
Input: problem
Output: solution, or failure

frontier ← Priority Queue
add starting node to frontier with priority = heuristic(start)  

while frontier is not empty do
    path ← remove node from frontier with lowest priority
    node ← path.node
    add node to explored set
    
    for each neighbor of node do
        if neighbor not in explored set then
            new_path ← Path(neighbor, path, edge)
            if neighbor is a goal node then
                return new_path as solution
            
            frontier.add((heuristic(neighbor), new_path))
            
return failure
```
**Code :** 
```bash
import heapq

def man_dist(curr, goal):
    goal = [val for i in goal for val in i]
    dist = 0
    for i in range(3):
        for j in range(3):
            if (curr[i][j] != 0):
                goal_index = goal.index(curr[i][j])
                x = goal_index // 3
                y = goal_index % 3
                dist += abs(i - x) + abs(j - y)
    return dist


def get_neighbours(state):
    for i, row in enumerate(state):
        for j, val in enumerate(row):
            if val == 0:
                x0, y0 = i, j
                break

    neighbours = []
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for (dx, dy) in moves:
        x, y = x0 + dx, y0 + dy
        if (0 <= x < 3 and 0 <= y < 3):
            new_state = [list(row) for row in state]
            new_state[x0][y0], new_state[x][y] = new_state[x][y], new_state[x0][y0]
            neighbours.append(new_state)

    return neighbours


def to_tup(state):
    return tuple(tuple(i) for i in state)

def find_path(state, parent):
    path = []
    while state:
        path.append(state)
        state = parent[to_tup(state)]
    return path[::-1]

def greedy_bfs(start, goal):
    pq = []
    heapq.heappush(pq, (man_dist(start, goal), start))
    explored = set()
    parent = {to_tup(start): None}

    while (pq):
        dist, curr = heapq.heappop(pq)
        if (curr == goal):
            return find_path(curr, parent)

        explored.add(to_tup(curr))
        for n in get_neighbours(curr):
            neighbour = to_tup(n)
            if neighbour not in explored:
                heapq.heappush(pq, (man_dist(n, goal), n))
                if neighbour not in parent:
                    parent[neighbour] = curr

    return None


def a_star(start, goal):
    pq = []
    heapq.heappush(pq, (man_dist(start, goal), 0, start))

    explored = set()
    parent = {to_tup(start): None}
    cost = {to_tup(start): 0}

    while pq:
        i, g_n, curr = heapq.heappop(pq)
        if (curr == goal):
            return find_path(curr, parent)

        explored.add(to_tup(curr))
        for n in get_neighbours(curr):
            neighbour = to_tup(n)
            new_gn = g_n + 1

            if neighbour not in explored or new_gn < cost.get(neighbour, float('inf')):
                cost[neighbour] = new_gn
                f_n = new_gn + man_dist(n, goal)
                heapq.heappush(pq, (f_n, new_gn, n))
                parent[neighbour] = curr

    return None


init_state = [[7,2,4], [5,0,6], [8,3,1]]
goal_state = [[0,1,2], [3,4,5], [6,7,8]]

sol1 = greedy_bfs(init_state, goal_state)
sol2 = a_star(init_state, goal_state)


if (sol1):
    print("\nGREEDY BEST FIRST SEARCH")
    for i in sol1:
        for j in i:
            for k in j:
                print(k,end=" ")
            print()
        print()
else:
    print("No solution found.")

if (sol2):
    print("\nA* SEARCH")
    for i in sol2:
        for j in i:
            for k in j:
                print(k,end=" ")
            print()
        print()
else:
    print("No solution found.")
```
**Testing :**
```bash
runfile('C:/Users/niran/Desktop/COLLEGE/3/AI/ai.py', wdir='C:/Users/niran/Desktop/COLLEGE/3/AI')
GREEDY BEST FIRST SEARCH
7 2 4 
5 0 6 
8 3 1 

7 2 4 
0 5 6 
8 3 1 

0 2 4 
7 5 6 
8 3 1 

2 0 4 
7 5 6 
8 3 1 

2 4 0 
7 5 6 
8 3 1 

2 4 6 
7 5 0 
8 3 1 

2 4 6 
7 0 5 
8 3 1 

2 4 6 
7 3 5 
8 0 1 

2 4 6 
7 3 5 
0 8 1 

2 4 6 
0 3 5 
7 8 1 

2 4 6 
3 0 5 
7 8 1 

2 0 6 
3 4 5 
7 8 1 

2 6 0 
3 4 5 
7 8 1 

2 6 5 
3 4 0 
7 8 1 

2 6 5 
3 4 1 
7 8 0 

2 6 5 
3 4 1 
7 0 8 

2 6 5 
3 0 1 
7 4 8 

2 0 5 
3 6 1 
7 4 8 

0 2 5 
3 6 1 
7 4 8 

3 2 5 
0 6 1 
7 4 8 

3 2 5 
6 0 1 
7 4 8 

3 2 5 
6 1 0 
7 4 8 

3 2 0 
6 1 5 
7 4 8 

3 0 2 
6 1 5 
7 4 8 

3 1 2 
6 0 5 
7 4 8 

3 1 2 
6 4 5 
7 0 8 

3 1 2 
6 4 5 
0 7 8 

3 1 2 
0 4 5 
6 7 8 

0 1 2 
3 4 5 
6 7 8 


A* SEARCH
7 2 4 
5 0 6 
8 3 1 

7 2 4 
0 5 6 
8 3 1 

0 2 4 
7 5 6 
8 3 1 

2 0 4 
7 5 6 
8 3 1 

2 5 4 
7 0 6 
8 3 1 

2 5 4 
7 3 6 
8 0 1 

2 5 4 
7 3 6 
0 8 1 

2 5 4 
0 3 6 
7 8 1 

2 5 4 
3 0 6 
7 8 1 

2 5 4 
3 6 0 
7 8 1 

2 5 0 
3 6 4 
7 8 1 

2 0 5 
3 6 4 
7 8 1 

0 2 5 
3 6 4 
7 8 1 

3 2 5 
0 6 4 
7 8 1 

3 2 5 
6 0 4 
7 8 1 

3 2 5 
6 4 0 
7 8 1 

3 2 5 
6 4 1 
7 8 0 

3 2 5 
6 4 1 
7 0 8 

3 2 5 
6 0 1 
7 4 8 

3 2 5 
6 1 0 
7 4 8 

3 2 0 
6 1 5 
7 4 8 

3 0 2 
6 1 5 
7 4 8 

3 1 2 
6 0 5 
7 4 8 

3 1 2 
6 4 5 
7 0 8 

3 1 2 
6 4 5 
0 7 8 

3 1 2 
0 4 5 
6 7 8 

0 1 2 
3 4 5 
6 7 8 

```
**Problem Description 2:**

You are given an 8-litre jar full of water and two empty jars of 5- and 3-litre capacity. You have to get exactly 4 litres of water in one of the jars. You can completely empty a jar into another jar with space or completely fill up a jar from another jar.

1. Formulate the problem: Identify states, actions, initial state, goal state(s). Represent the state by a 3-tuple. For example, the intial state state is (8,0,0). (4,1,3) is a goal state
(there may be other goal states also).
2. Use a suitable data structure to keep track of the parent of every state. Write a function to print the sequence of states and actions from the initial state to the goal state.
3. Write a function next states(s) that returns a list of successor states of a given state s.
4. Implement Breadth-First-Search algorithm to search the state space graph for a goal state that produces the required sequence of pourings. Use a Queue as frontier that stores the discovered states yet be explored. Use a dictionary for explored that is used to store the explored states.
5. Modify your program to trace the contents of the Queue in your algorithm. How many
states are explored by your algorithm?




**Algorithm:**
```bash
Input: problem
Output: solution, or failure
frontier ← Queue
add starting node to frontier
parent[start] ← None

while frontier is not empty do
    path ← remove node from frontier
    node ← path.node
    add node to explored set
    
    for each neighbor of node do
        if neighbor not in explored set then
            new_path ← Path(neighbor, path)
            if neighbor is a goal node then
                return new_path as solution
            
            frontier.append(new_path)
            
return failure
```
**Code :** 
```python
from collections import deque

def next_states(curr_state):
    x, y, z = curr_state
    states = []

    val = min(x, 5 - y)
    states.append((x - val, y + val, z))
    
    val = min(x, 3 - z)
    states.append((x - val, y, z + val))

    val = min(y, 8 - x)
    states.append((x + val, y - val, z))
    
    val = min(y, 3 - z)
    states.append((x, y - val, z + val))
    
    val = min(z, 8 - x)
    states.append((x + val, y, z - val))

    val = min(z, 5 - y)
    states.append((x, y + val, z - val))
    
    return states


def display_path(state, explored):
    path = []
    while state is not None:
        path.append(state)
        state = explored[state]
    
    path.reverse()
    
    print("\nSEQUENCE OF STATES")
    for s in path:
        print(s)


def bfs(initial_state):
    queue = deque([initial_state])
    explored = {initial_state: None}  

    while queue:
        state = queue.popleft()

        if state[0] == 4 or state[1] == 4 or state[2] == 4:
            display_path(state, explored)
            return explored

        for next_state in next_states(state):
            if next_state not in explored:
                queue.append(next_state)
                explored[next_state] = state

    print("\nNo solution found.")
    return explored


initial = (8, 0, 0)
exp = bfs(initial)
print("\nNumber of states explored = ", len(exp)-1)


```

**Testing :**
```bash
runfile('C:/Users/niran/Desktop/COLLEGE/3/AI/ai.py', wdir='C:/Users/niran/Desktop/COLLEGE/3/AI')

SEQUENCE OF STATES
(8, 0, 0)
(3, 5, 0)
(3, 2, 3)
(6, 2, 0)
(6, 0, 2)
(1, 5, 2)
(1, 4, 3)

Number of states explored =  13 
```

<div style="page-break-after: always;"></div>


## Assignment 4
**Date :** 29-08-2024

**Problem Description :**

Place 8 queens “safely” in a 8×8 chessboard – no queen is under attack from any other queen
(in horizontal, vertical and diagonal directions). Formulate it as a constraint satisfaction problem.
- One queen is placed in each column.
- Variables are the rows in which queens are placed in the columns
- Assignment: 8 row indexes.
- Evaluation function: the number of attacking pairs in 8-queens
Implement a local search algorithm to find one safe assignment.

**Algorithm:**
1. Local Search
```bash
Input: problem
Output: solution, or failure

current ← initial state of problem
while true do
    neighbors ← generate neighbors of current
    best_neighbor ← find the best state in neighbors

    if best_neighbor is better than current then
        current ← best_neighbor
    else
        return current as solution
```
2. Stochastic Search
```bash
Input: problem
Output: solution, or failure

current ← initial solution of problem
while stopping criteria not met do
    if current is a valid solution then
        return current as solution
    
    neighbor ← randomly select a neighbor of current
    neighbor_value ← evaluate(neighbor)

    if neighbor_value < evaluate(current) then
        current ← neighbor
    else
        if random() < acceptance_probability(current, neighbor_value) then
            current ← neighbor
return failure
```
**Code :** 
1. Local Search
```python
import random

def init(n):
    l = list(range(n))
    random.shuffle(l)
    return tuple(l)

def neighbors(state, n):
    lst = []
    for i in range(n):
        for j in range(i + 1, n):
            new = state[:i] + (state[j],) + state[i + 1:j] + (state[i],) + state[j + 1:]
            lst.append(new)
    return lst

def evaluate(state, n):
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == abs(state[i] - state[j]):
                c += 1
    return c

def print_board(state):
    n = len(state)
    board = [['.' for _ in range(n)] for _ in range(n)]
    for row, col in enumerate(state):
        board[row][col] = 'Q'
    for row in board:
        print(" ".join(row))
    print("\n")

def local_search(n):
    cur_state = init(n)
    cur_val = evaluate(cur_state, n)
    print("Initial board:")
    print_board(cur_state)
    while cur_val > 0:
        xx = neighbors(cur_state, n)
        for i in xx:
            x = evaluate(i, n)
            if x < cur_val:
                cur_val = x
                cur_state = i
                print(f"Current state with evaluation {cur_val}:")
                print_board(cur_state)
                break
        else:
            print("\nRandom Restart!\n")
            return local_search(n)
            break
    print("Solution found:")
    print_board(cur_state)
    return cur_state

n = int(input("No. of rows = "))
local_search(n)
```
2. Stochastic Search
```python
import random

def no_attacking_pairs(board):
    """ Count the number of pairs of queens that are attacking each other. """
    n = len(board)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (board[i] == board[j] or
                abs(board[i] - board[j]) == abs(i - j)):
                count += 1
    return count

def possible_successors(conf):
    n = len(conf)
    state_value = {}

    for i in range(n):
        for j in range(n):
            if j != conf[i]: 
                x = conf[:i] + [j] + conf[i + 1:]
                ap = no_attacking_pairs(x)
                state_value[ap] = x

    min_conflicts = min(state_value.keys())
    return state_value[min_conflicts], min_conflicts

def print_board(board):
    """ Display the board with queens as 'Q' and empty spaces as '.' """
    n = len(board)
    for row in range(n):
        line = ""
        for col in range(n):
            if board[row] == col:
                line += "Q "
            else:
                line += ". "
        print(line)
    print("\n")

def random_restart(n):
    global iteration
    iteration += 1
    print(f"\nRandom Restart #{iteration}")
    l = [random.randint(0, n - 1) for _ in range(n)]
    print_board(l)
    return l

def eight_queens(initial):
    conflicts = no_attacking_pairs(initial)
    print("Initial configuration:")
    print_board(initial)
    
    while conflicts > 0:
        new, new_conflicts = possible_successors(initial)
        if new_conflicts < conflicts:
            conflicts = new_conflicts
            initial = new
            print("New configuration with fewer conflicts:")
            print_board(initial)
        else:
            initial = random_restart(len(initial))
            conflicts = no_attacking_pairs(initial)
    
    print("Solution found:")
    print_board(initial)
    return initial

iteration = 0
n = int(input('No. of rows = '))
board = random_restart(n)

solution = eight_queens(board)
print("Number of random restarts =", iteration)
print("Final configuration of the board =")
print_board(solution)
```
**Testing :**
1. Local Search
```bash
runfile('C:/Users/niran/Desktop/COLLEGE/3/AI/ai.py', wdir='C:/Users/niran/Desktop/COLLEGE/3/AI')
No. of rows = 4
Initial board:
. . . Q
Q . . .
. Q . .
. . Q .


Current state with evaluation 1:
Q . . .
. . . Q
. Q . .
. . Q .


Current state with evaluation 0:
. Q . .
. . . Q
Q . . .
. . Q .


Solution found:
. Q . .
. . . Q
Q . . .
. . Q .


```
2. Stochastic Search
```bash
runfile('C:/Users/niran/Desktop/COLLEGE/3/AI/ai.py', wdir='C:/Users/niran/Desktop/COLLEGE/3/AI')
No. of rows = 4

Random Restart #1
. . Q .
. . . Q
. . . Q
. Q . .


Initial configuration:
. . Q .
. . . Q
. . . Q
. Q . .


New configuration with fewer conflicts:
. . Q .
Q . . .
. . . Q
. Q . .


Solution found:
. . Q .
Q . . .
. . . Q
. Q . .


Number of random restarts = 1
Final configuration of the board =
. . Q .
Q . . .
. . . Q
. Q . .


```
<div style="page-break-after: always;"></div>


## Assignment 5
**Date :** 05-09-2024

**Problem Description :**

1. Class Variable
Define a class Variable consisting of a name and a domain. The domain of a variable is a list or a tuple, as the ordering will matter in the representation of constraints. We would like to create a Variable object, for example, as
X = Variable(’X’, {1,2,3})

2. Class Constraint
Define a class Constraint consisting of
- A tuple (or list) of variables called the scope.
- A condition, a Boolean function that takes the same number of arguments as there are
variables in the scope. The condition must have a name property that gives a printable
name of the function; built-in functions and functions that are defined using def have such
a property; for other functions you may need to define this property.
- An optional name
We would like to create a Variable object, for example, as Constraint([X,Y],lt) where lt is
a function that tests whether the first argument is less than the second one.
Add the following methods to the class.
def can_evaluate(self, assignment):
"""
assignment is a variable:value dictionary
returns True if the constraint can be evaluated given assignment
"""
def holds(self,assignment):
"""returns the value of Constraint evaluated in assignment.
precondition: all variables are assigned in assignment, ie self.can_evaluate(assignment) """

3. Class CSP
A constraint satisfaction problem (CSP) requires:
- variables: a list or set of variables
- constraints: a set or list of constraints.
Other properties are inferred from these:
- var to const is a mapping fromvariables to set of constraints, such that var to const[var]
is the set of constraints with var in the scope.
Add a method consistent(assignment) to class CSP that returns true if the assignment is consistent with each of the constraints in csp (i.e., all of the constraints that can be evaluated
evaluate to true).

We may create a CSP problem, for example, as

X = Variable(’X’, {1,2,3})
Y = Variable(’Y’, {1,2,3})
Z = Variable(’Z’, {1,2,3})
csp0 = CSP("csp0", {X,Y,Z},
[Constraint([X,Y],lt),
Constraint([Y,Z],lt)])

The CSP csp0 has variables X, Y and Z, each with domain {1, 2, 3}. The con straints are X < Y and Y < Z.

4. 8-Queens
Place 8 queens “safely” in a 8×8 chessboard – no queen is under attack from any other queen (in horizontal, vertical and diagonal directions). Formulate it as a constraint satisfaction problem.
- One queen is placed in each column.
- Variables are the rows in which queens are placed in the columns
- Assignment: 8 row indexes.
Represent it as a CSP.

5. Simple DFS Solver
Solve CSP using depth-first search through the space of partial assignments. This takes in a CSP problem and an optional variable ordering (a list of the variables in the CSP). It returns a generator of the solutions.


**Algorithm:**
```bash
Input: assignment, CSP (Constraint Satisfaction Problem)
Output: solution, or failure

function backtrack(assignment, csp):
    if length of assignment equals number of csp variables then
        return assignment

    unassigned ← variables in csp not in assignment
    var ← first variable in unassigned

    for each value in var.domain do
        new_assignment ← copy of assignment
        new_assignment[var] ← value

        if csp is consistent with new_assignment then
            result ← backtrack(new_assignment, csp)
            if result is not None then
                return result

    return None
```
**Code :** 
```python
import random

class Variable:
    def __init__(self, name, domain):
        self.name = name
        self.domain = list(domain)

    def __repr__(self):
        return f"{self.name}: {self.domain}"

class Constraint:
    def __init__(self, scope, condition, name=None):
        self.scope = scope
        self.condition = condition
        self.name = name or condition.__name__

    def can_evaluate(self, assignment):
        return all(var in assignment for var in self.scope)

    def holds(self, assignment):
        if not self.can_evaluate(assignment):
            raise ValueError("Cannot evaluate constraint: missing variable assignments.")
        values = [assignment[var] for var in self.scope]
        return self.condition(*values)

    def __repr__(self):
        return f"Constraint({self.scope}, {self.name})"

class CSP:
    def __init__(self, name, variables, constraints):
        self.name = name
        self.variables = variables
        self.constraints = constraints
        self.var_to_const = {var: set() for var in variables}
        
        for constraint in constraints:
            for var in constraint.scope:
                self.var_to_const[var].add(constraint)

    def consistent(self, assignment):
        return all(constraint.holds(assignment) for constraint in self.constraints if constraint.can_evaluate(assignment))

    def __repr__(self):
        return f"CSP({self.name}) with variables {self.variables} and constraints {self.constraints}"

def dfs_solver(csp, var_order=None):
    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            yield dict(assignment)
            return
        
        unassigned_vars = [v for v in var_order if v not in assignment]
        if not unassigned_vars:
            return
        var = unassigned_vars[0]
        
        for value in var.domain:
            assignment[var] = value
            if csp.consistent(assignment):
                yield from backtrack(assignment)
            assignment.pop(var)
    
    var_order = var_order or list(csp.variables)
    yield from backtrack({})

def not_under_attack(row1, col1, row2, col2):
    return row1 != row2 and abs(row1 - row2) != abs(col1 - col2)

def create_n_queens_csp(n):
    columns = [Variable(f"Q{i}", range(n)) for i in range(n)]
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(Constraint([columns[i], columns[j]], lambda r1, r2, i=i, j=j: not_under_attack(r1, i, r2, j)))
    return CSP(f"{n}-Queens", columns, constraints)

def display_solution(solution, n):
    board = [[0] * n for _ in range(n)]
    for var, row in solution.items():
        col = int(var.name[1:])
        board[row][col] = 1
    for row in board:
        print(" ".join(str(cell) for cell in row))
    print()

n = int(input("No. of queens = "))
csp_n_queens = create_n_queens_csp(n)

solutions = list(dfs_solver(csp_n_queens, csp_n_queens.variables))
print(f"Number of solutions found = {len(solutions)}")

for idx, solution in enumerate(solutions[:2], 1):  # Display the first two solutions
    print(f"Solution {idx}:")
    display_solution(solution, n)
```
**Testing :**
```bash
runfile('C:/Users/niran/Desktop/COLLEGE/3/AI/ai.py', wdir='C:/Users/niran/Desktop/COLLEGE/3/AI')
No. of queens = 8
Number of solutions found = 92
Solution 1:
1 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0
0 0 0 0 1 0 0 0
0 0 0 0 0 0 0 1
0 1 0 0 0 0 0 0
0 0 0 1 0 0 0 0
0 0 0 0 0 1 0 0
0 0 1 0 0 0 0 0

Solution 2:
1 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0
0 0 0 1 0 0 0 0
0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 1
0 1 0 0 0 0 0 0
0 0 0 0 1 0 0 0
0 0 1 0 0 0 0 0

```

## Assignment 6
**Date :** 05-09-2024

**Problem Description :**

Consider two-player zero-sum games, where a player only wins when another player loses. This can be modeled with a single utility which one agent (the maximizing agent) is trying maximize and the other agent (the minimizing agent) is trying to minimize. Define a class Node to represent a node in a game tree.

class Node(Displayable):
"""A node in a search tree. It has a name a string isMax is True if it is a maximizing node, otherwise it is minimizing node children is the list of children value is what it evaluates to if it is a leaf.
"""
Create the game tree given below:
1. Implement minimax algorithm for a zero-sum two player game as a function minimax(node,
depth). Let minimax(node, depth) return both the score and the path. Test it on the
game tree you have created.
2. Modify the minimax function to include αβ-pruning.


**Algorithm:**
1. Minimax
```bash
function minimax(node):
    if node is a leaf then
        return evaluate(node), None

    if node is a maximizing node then
        max_score ← -∞
        max_path ← None
        for each child in node.children() do
            score, path ← minimax(child)
            if score > max_score then
                max_score ← score
                max_path ← (child.name, path)

        return max_score, max_path

    else  // node is a minimizing node
        min_score ← ∞
        min_path ← None
        for each child in node.children() do
            score, path ← minimax(child)
            if score < min_score then
                min_score ← score
                min_path ← (child.name, path)

        return min_score, min_path
```
2. Without Alpha-beta pruning
```bash
function minimax(node, alpha, beta):
    if node is a leaf then
        return evaluate(node), None

    if node is a maximizing node then
        max_path ← None
        for each child in node.children() do
            score, path ← minimax(child, alpha, beta)
            if score >= beta then
                return score, None 
            if score > alpha then
                alpha ← score
                max_path ← (child.name, path)
        return alpha, max_path

    else  // node is a minimizing node
        min_path ← None
        for each child in node.children() do
            score, path ← minimax(child, alpha, beta)
            if score <= alpha then
                return score, None  
            if score < beta then
                beta ← score
                min_path ← (child.name, path)
        return beta, min_path
```
**Code :** 
```python
class Node:
    def __init__(self,name,isMax,value,children):
        self.name=name
        self.value=value
        self.isMax=isMax
        self.allChildren=children

    def isLeaf(self):
        return self.allChildren is None
    
    def children(self):
        return self.allChildren
    
    def evaluate(self):
        return self.value
    
class MiniMax:
    def minimax(self,node,depth):
        count=0
        if node.isLeaf():
            return node.evaluate(),None,1
        elif node.isMax:
            max_score=float("-inf")
            max_path=None
            for c in node.children():
                score,path,c=self.minimax(c,depth+1)
                count+=c
                if score>max_score:
                    max_score=score
                    max_path=node.name,path
            return max_score,max_path,count+1
        else:
            min_score=float("inf")
            min_path=None
            for c in node.children():
                score,path,c=self.minimax(c,depth+1)
                count+=c
                if score<min_score:
                    min_score=score
                    min_path=node.name,path
            return min_score,min_path,count+1
        
    def minimaxAB(self,node,alpha,beta,depth):
        count=0
        best=None
        if node.isLeaf():
            return node.evaluate(),None,1
        elif node.isMax:
            for c in node.children():
                score,path,c=self.minimaxAB(c,alpha,beta,depth+1)
                count+=c
                if score>=beta:
                    return score,None,1
                if score>alpha:
                    alpha=score
                    best=node.name,path
            return alpha,best,count+1
        else:
            for c in node.children():
                score,path,c=self.minimaxAB(c,alpha,beta,depth+1)
                count+=c
                if score<=alpha:
                    return score,None,1
                if score<beta:
                    beta=score
                    best=node.name,path
            return beta,best,count+1


n16=Node("16",None,20,None)
n17=Node("17",None,float("inf"),None)
n18=Node("18",None,-10,None)
n19=Node("19",None,9,None)
n20=Node("20",None,-8,None)
n21=Node("21",None,8,None)
n22=Node("22",None,8,None)
n23=Node("23",None,6,None)
n24=Node("24",None,float("inf"),None)
n25=Node("25",None,-10,None)
n26=Node("26",None,-5,None)


n8=Node("8",False,None,[n16,n17])
n9=Node("9",False,None,[n18,n19])
n10=Node("10",False,float("-inf"),None)
n11=Node("11",False,None,[n20,n21])
n12=Node("12",False,None,[n22])
n13=Node("13",False,None,[n23,n24])
n14=Node("14",False,float("-inf"),None)
n15=Node("15",False,None,[n25,n26])

#max nodes
n4=Node("4",True,None,[n8,n9])
n5=Node("5",True,None,[n10,n11])
n6=Node("6",True,None,[n12,n13])
n7=Node("7",True,None,[n14,n15])

#min nodes
n2=Node("2",False,None,[n4,n5])
n3=Node("3",False,None,[n6,n7])

#root
n1=Node("1",True,None,[n2,n3])

Solver=MiniMax()
score,path,count=Solver.minimax(n1,0)
print(f"Score = {score}\nPath = {path}\nNo. of nodes explored = {count}")

score,path,count=Solver.minimaxAB(n1,float("-inf"),float("inf"),0)
print(f"Score = {score}\nPath = {path}\nNo. of nodes explored = {count}")
```


**Testing :**
```bash
runfile('C:/Users/niran/Desktop/COLLEGE/3/AI/ai.py', wdir='C:/Users/niran/Desktop/COLLEGE/3/AI')
Score = -8
Path = ('1', ('2', ('5', ('11', None))))
No. of nodes explored = 26
Score = -8
Path = ('1', ('2', ('5', ('11', None))))
No. of nodes explored = 13
PS C:\Users\niran\Downloads> 
```


<div style="page-break-after: always;"></div>

## Assignment 7
**Date :** 25/09/2024

**Problem Description :**

1 Knowledge Base

Define a class for Clause. A clause consists of a head (an atom) and a body. A body is represented as a list of atoms. Atoms are represented as strings.
class Clause(object):
"""A definite clause"""
def __init__(self,head,body=[]):
"""clause with atom head and lost of atoms body"""
self.head=head
self.body = body

Define a class Askable to represent atoms askable from the user.
class Askable(object):
"""An askable atom"""
def __init__(self,atom):
"""clause with atom head and lost of atoms body"""
self.atom=atom

Define a class KB to represent a knowldege base. A knowledge base is a list of clauses and askables. In order to make top-down inference faster, create a dictionary that maps each atom into the set of clauses with that atom in the head.

class KB(Displayable):
"""A knowledge base consists of a set of clauses.
This also creates a dictionary to give fast access to
the clauses with an atom in head.
1
"""
def __init__(self, statements=[]):
self.statements = statements
self.clauses = ...
self.askables = ...
self.atom_to_clauses = {}
...
def add_clause(self, c):
...
def clauses_for_atom(self,a):
...

With Clause and KB classes, we can define a trivial example KB as shown below:
triv_KB = KB([
Clause(’i_am’, [’i_think’]),
Clause(’i_think’),
Clause(’i_smell’, [’i_exist’])
])

Represent the electrical domain of Example 5.8 of Poole and Macworth.

2 Proof Procedures

1. Implement a bottom-up proof procedure for definite clauses in PL to compute the fixed point consequence set of a knowledge base.
2. Implement a top-down proof procedure prove(kb, goal) for definite clauses in PL. It
takes kb, a knowledge base KB and goal as inputs, where goal is a list of atoms. It returns
True if kb ⊢ goal.


**Algorithm:**
1. Top-Down Approach
```bash
function prove(KB, ans_body, indent=""):
    print(indent + 'yes <- ' + join(ans_body with " & "))

    if ans_body is not empty then
        selected ← ans_body[0]
        
        if selected is an askable in KB then
            ask user if selected is true
            if user confirms selected is true then
                return prove(KB, ans_body[1:], indent + " ")
            else
                return False
        
        else
            for each clause in KB.clauses_for_atom(selected) do
                if prove(KB, clause.body + ans_body[1:], indent + " ") then
                    return True

            return False

    else  
        return True
```
2. Bottom-Up Approach
```bash
function fixed_point(KB):
    fp ← ask_askables(KB)
    added ← True

    while added do
        added ← False  // Indicates if an atom was added this iteration

        for each clause in KB.clauses do
            if clause.head is not in fp and all elements of clause.body are in fp then
                add clause.head to fp
                added ← True
                print(clause.head, "added to fixed point due to clause:", clause)

    return fp
```
**Code :** 
```python

class Clause():
	def __init__(self,head,body=[]):
		self.head = head
		self.body = body
		
class Askable():
	def __init__(self,atom):
		self.atom = atom

class KB:
	def __init__(self, statements=[]):
		self.statements = statements
		self.clauses = [c for c in statements if isinstance(c, Clause)]
		self.askables = [c.atom for c in statements if isinstance(c, Askable)]
		self.atom_to_clauses = {}
		for c in self.clauses:
			self.add_clause(c)
			
	def add_clause(self, c):
		if c.head in self.atom_to_clauses:
			self.atom_to_clauses[c.head].append(c)
		else:
			self.atom_to_clauses[c.head] = [c]

	def clauses_for_atom(self,a):
		if a in self.atom_to_clauses:
			return self.atom_to_clauses[a]
		else:
			return []
triv_KB = KB([
Clause('i_am', ['i_think']),
Clause('i_think'),
Clause('i_smell', ['i_exist'])
])

elect = KB([
Clause('light_l1'),
Clause('light_l2'),
Clause('ok_l1'),
Clause('ok_l2'),
Clause('ok_cb1'),
Clause('ok_cb2'),
Clause('live_outside'),
Clause('live_l1', ['live_w0']),
Clause('live_w0', ['up_s2','live_w1']),
Clause('live_w0', ['down_s2','live_w2']),
Clause('live_w1', ['up_s1', 'live_w3']),
Clause('live_w2', ['down_s1','live_w3']),
Clause('live_l2', ['live_w4']),
Clause('live_w4', ['up_s3','live_w3']),
Clause('live_p_1', ['live_w3']),
Clause('live_w3', ['live_w5', 'ok_cb1']),
Clause('live_p_2', ['live_w6']),
Clause('live_w6', ['live_w5', 'ok_cb2']),
Clause('live_w5', ['live_outside']),
Clause('lit_l1', ['light_l1', 'live_l1', 'ok_l1']),
Clause('lit_l2', ['light_l2', 'live_l2', 'ok_l2']),
Askable('up_s1'),
Askable('down_s1'),
Askable('up_s2'),
Askable('down_s2'),
Askable('up_s3'),
Askable('down_s2')
])

def bottom_up_proof(kb):
    derived = set() 
    changed = True

    while changed:
        changed = False
        for clause in kb.clauses:
            if all(body_atom in derived for body_atom in clause.body):
                if clause.head not in derived:
                    derived.add(clause.head)
                    changed = True

    return derived
    
def prove(kb, goal):
    return prove_all(kb, goal, set())

def prove_all(kb, goals, proved):
    if not goals:
        return True
    first, rest = goals[0], goals[1:]
    
    if first in proved:
        return prove_all(kb, rest, proved)

    for clause in kb.clauses_for_atom(first):
        if prove_all(kb, clause.body + rest, proved | {first}):
            return True

    return False    
    
# Bottom-up inference
derived_atoms = bottom_up_proof(triv_KB)
print("Derived atoms (bottom-up):", derived_atoms)
print()

# Top-down inference
goal = ['i_am']
result = prove(triv_KB, goal)
print(f"Can we prove {goal}? {result}")


```
**Testing :**
```bash
runfile('C:/Users/niran/Desktop/COLLEGE/3/AI/ai.py', wdir='C:/Users/niran/Desktop/COLLEGE/3/AI')
Derived atoms (bottom-up): {'i_am', 'i_think'}

Can we prove ['i_am']? True
```


<div style="page-break-after: always;"></div>

## Assignment 8
**Date :** 25-09-2024

**Problem Description :**

Inference using Bayesian Network (BN) – Joint Probability Distribution
The given Bayesian Network has 5 variables with the dependency between the variables as shown below:
 
1. The marks (M) of a student depends on:
- Exam level (E): This is a discrete variable that can take two values, (difficult, easy) and
- IQ of the student (I): A discrete variable that can take two values (high, low)
2. The marks (M) will, in turn, predict whether he/she will get admitted (A) to a university.
3. The IQ (I) will also predict the aptitude score (S) of the student.

Write functions to

1. Construct the given DAG representation using appropriate libraries.
2. Read and print the Conditional Probability Table (CPT) for each variable.
3. Calculate the joint probability distribution of the BN using 5 variables.
Observation: Write the formula for joint probability distribution and explain each parameter.
Justify the answer with the advantage of BN.

**Algorithm:**
```bash
Input: Prior probabilities for hypotheses H
Output: Posterior Probability P(H|E)

function bayes_algorithm(P_H, P_E_given_H):
    P_E ← 0
    for each hypothesis H in P_H:
        P_E ← P_E + P(E | H) * P(H)  \

    for each hypothesis H in P_H:
        P_H_given_E[H] ← (P(E | H) * P(H)) / P_E   

    return P_H_given_E
```
**Code :** 
```python
P_e = {0: 0.7, 1: 0.3}  
P_i = {0: 0.8, 1: 0.2}  

P_m_given_e_i = {
    (0, 0): {0: 0.6, 1: 0.4},
    (0, 1): {0: 0.1, 1: 0.9},
    (1, 0): {0: 0.5, 1: 0.5},
    (1, 1): {0: 0.2, 1: 0.8}
}  

P_a_given_m = {
    0: {0: 0.6, 1: 0.4},
    1: {0: 0.9, 1: 0.1}
}  

P_s_given_i = {
    0: {0: 0.75, 1: 0.25},
    1: {0: 0.4, 1: 0.6}
}  

def print_cpd_exam_level():
    print("CPD for Exam Level (e):")
    print("+----------------+------+")
    print("| e              | P(e) |")
    print("+----------------+------+")
    for e_state, prob in P_e.items():
        print(f"| {e_state:<14} | {prob:<4} |")
    print("+----------------+------+\n")

def print_cpd_iq():
    print("CPD for IQ (i):")
    print("+----------------+------+")
    print("| i              | P(i) |")
    print("+----------------+------+")
    for i_state, prob in P_i.items():
        print(f"| {i_state:<14} | {prob:<4} |")
    print("+----------------+------+\n")

def print_cpd_marks():
    print("CPD for Marks (m):")
    print("+----------------+----------------+----------------+----------------+")
    print("| e              | i              | P(m=0)        | P(m=1)        |")
    print("+----------------+----------------+----------------+----------------+")
    for (e_state, i_state), m_probs in P_m_given_e_i.items():
        print(f"| {e_state:<14} | {i_state:<14} | {m_probs[0]:<14} | {m_probs[1]:<14} |")
    print("+----------------+----------------+----------------+----------------+\n")

def print_cpd_admission():
    print("CPD for Admission (a):")
    print("+----------------+----------------+----------------+")
    print("| m              | P(a=0)        | P(a=1)        |")
    print("+----------------+----------------+----------------+")
    for m_state, a_probs in P_a_given_m.items():
        print(f"| {m_state:<14} | {a_probs[0]:<14} | {a_probs[1]:<14} |")
    print("+----------------+----------------+----------------+\n")

def print_cpd_aptitude_score():
    print("CPD for Aptitude Score (s):")
    print("+----------------+----------------+----------------+")
    print("| i              | P(s=0)        | P(s=1)        |")
    print("+----------------+----------------+----------------+")
    for i_state, s_probs in P_s_given_i.items():
        print(f"| {i_state:<14} | {s_probs[0]:<14} | {s_probs[1]:<14} |")
    print("+----------------+----------------+----------------+\n")

def calculate_jpd(e_state, i_state, m_state, a_state, s_state):
    P_e_val = P_e[e_state]
    P_i_val = P_i[i_state]
    P_m_val = P_m_given_e_i[(e_state, i_state)][m_state]
    P_a_val = P_a_given_m[m_state][a_state]
    P_s_val = P_s_given_i[i_state][s_state]
    jpd = P_e_val * P_i_val * P_m_val * P_a_val * P_s_val
    return jpd

def print_jpd_table():
    print("Joint Probability Distribution Table:")
    print("+----------------+----------------+----------------+----------------+----------------+----------------+")
    print("| e              | i              | m              | a              | s              | P(e, i, m, a, s)|")
    print("+----------------+----------------+----------------+----------------+----------------+----------------+")
    for e_state in P_e.keys():
        for i_state in P_i.keys():
            for m_state in [0, 1]:
                for a_state in [0, 1]:
                    for s_state in [0, 1]:
                        jpd = calculate_jpd(e_state, i_state, m_state, a_state, s_state)
                        print(f"| {e_state:<14} | {i_state:<14} | {m_state:<14} | {a_state:<14} | {s_state:<14} | {jpd:<14.4f} |")
    print("+----------------+----------------+----------------+----------------+----------------+----------------+")

def print_jpd_formula():
    print("Joint Probability Distribution Formula:")
    print("P(e, i, m, a, s) = P(e) * P(i) * P(m | e, i) * P(a | m) * P(s | i)\n")
    print("Where:")
    print(" P(e): Probability of Exam Level")
    print(" P(i): Probability of IQ")
    print(" P(m | e, i): Probability of Marks given Exam Level and IQ")
    print(" P(a | m): Probability of Admission given Marks")
    print(" P(s | i): Probability of Aptitude Score given IQ\n")

def get_input_and_print_probability():
    print("Enter the states for the following variables (leave blank for unknown):")
    e_state = input("Exam Level (e) [0=easy/1=difficult]: ").strip() or None
    i_state = input("IQ (i) [0=low/1=high]: ").strip() or None
    m_state = input("Marks (m) [0=low/1=high]: ").strip() or None
    a_state = input("Admission (a) [0=no/1=yes]: ").strip() or None
    s_state = input("Aptitude Score (s) [0=poor/1=good]: ").strip() or None

    e_state = int(e_state) if e_state is not None else None
    i_state = int(i_state) if i_state is not None else None
    m_state = int(m_state) if m_state is not None else None
    a_state = int(a_state) if a_state is not None else None
    s_state = int(s_state) if s_state is not None else None

    valid_states_e = list(P_e.keys())
    valid_states_i = list(P_i.keys())
    valid_states_m = [0, 1]
    valid_states_a = [0, 1]
    valid_states_s = [0, 1]

    states_to_check = {
        'e': valid_states_e if e_state is None else [e_state],
        'i': valid_states_i if i_state is None else [i_state],
        'm': valid_states_m if m_state is None else [m_state],
        'a': valid_states_a if a_state is None else [a_state],
        's': valid_states_s if s_state is None else [s_state],
    }

    total_jpd = 0
    print("\nCalculating JPD for the following combinations:")
    for e in states_to_check['e']:
        for i in states_to_check['i']:
            for m in states_to_check['m']:
                for a in states_to_check['a']:
                    for s in states_to_check['s']:
                        jpd = calculate_jpd(e, i, m, a, s)
                        total_jpd += jpd
                        print(f"P(e={e}, i={i}, m={m}, a={a}, s={s}) = {jpd:.4f}")

    print(f"\nTotal Joint Probability for the given states = {total_jpd:.4f}")

print_cpd_exam_level()
print_cpd_iq()
print_cpd_marks()
print_cpd_admission()
print_cpd_aptitude_score()

print_jpd_table()
print_jpd_formula()

get_input_and_print_probability()
```
**Testing :**
```bash
runfile('C:/Users/niran/Desktop/COLLEGE/3/AI/ai.py', wdir='C:/Users/niran/Desktop/COLLEGE/3/AI')
CPD for Exam Level (e):
+----------------+------+
| e              | P(e) |
+----------------+------+
| 0              | 0.7  |
| 1              | 0.3  |
+----------------+------+

CPD for IQ (i):
+----------------+------+
| i              | P(i) |
+----------------+------+
| 0              | 0.8  |
| 1              | 0.2  |
+----------------+------+

CPD for Marks (m):
+----------------+----------------+----------------+----------------+
| e              | i              | P(m=0)        | P(m=1)        |
+----------------+----------------+----------------+----------------+
| 0              | 0              | 0.6            | 0.4            |
| 0              | 1              | 0.1            | 0.9            |
| 1              | 0              | 0.5            | 0.5            |
| 1              | 1              | 0.2            | 0.8            |
+----------------+----------------+----------------+----------------+

CPD for Admission (a):
+----------------+----------------+----------------+
| m              | P(a=0)        | P(a=1)        |
+----------------+----------------+----------------+
| 0              | 0.6            | 0.4            |
| 1              | 0.9            | 0.1            |
+----------------+----------------+----------------+

CPD for Aptitude Score (s):
+----------------+----------------+----------------+
| i              | P(s=0)        | P(s=1)        |
+----------------+----------------+----------------+
| 0              | 0.75           | 0.25           |
| 1              | 0.4            | 0.6            |
+----------------+----------------+----------------+

Joint Probability Distribution Table:
+----------------+----------------+----------------+----------------+----------------+----------------+
| e              | i              | m              | a              | s              | P(e, i, m, a, s)|
+----------------+----------------+----------------+----------------+----------------+----------------+
| 0              | 0              | 0              | 0              | 0              | 0.1512         |
| 0              | 0              | 0              | 0              | 1              | 0.0504         |
| 0              | 0              | 0              | 1              | 0              | 0.1008         |
| 0              | 0              | 0              | 1              | 1              | 0.0336         |
| 0              | 0              | 1              | 0              | 0              | 0.1512         |
| 0              | 0              | 1              | 0              | 1              | 0.0504         |
| 0              | 0              | 1              | 1              | 0              | 0.0168         |
| 0              | 0              | 1              | 1              | 1              | 0.0056         |
| 0              | 1              | 0              | 0              | 0              | 0.0034         |
| 0              | 1              | 0              | 0              | 1              | 0.0050         |
| 0              | 1              | 0              | 1              | 0              | 0.0022         |
| 0              | 1              | 0              | 1              | 1              | 0.0034         |
| 0              | 1              | 1              | 0              | 0              | 0.0454         |
| 0              | 1              | 1              | 0              | 1              | 0.0680         |
| 0              | 1              | 1              | 1              | 0              | 0.0050         |
| 0              | 1              | 1              | 1              | 1              | 0.0076         |
| 1              | 0              | 0              | 0              | 0              | 0.0540         |
| 1              | 0              | 0              | 0              | 1              | 0.0180         |
| 1              | 0              | 0              | 1              | 0              | 0.0360         |
| 1              | 0              | 0              | 1              | 1              | 0.0120         |
| 1              | 0              | 1              | 0              | 0              | 0.0810         |
| 1              | 0              | 1              | 0              | 1              | 0.0270         |
| 1              | 0              | 1              | 1              | 0              | 0.0090         |
| 1              | 0              | 1              | 1              | 1              | 0.0030         |
| 1              | 1              | 0              | 0              | 0              | 0.0029         |
| 1              | 1              | 0              | 0              | 1              | 0.0043         |
| 1              | 1              | 0              | 1              | 0              | 0.0019         |
| 1              | 1              | 0              | 1              | 1              | 0.0029         |
| 1              | 1              | 1              | 0              | 0              | 0.0173         |
| 1              | 1              | 1              | 0              | 1              | 0.0259         |
| 1              | 1              | 1              | 1              | 0              | 0.0019         |
| 1              | 1              | 1              | 1              | 1              | 0.0029         |
+----------------+----------------+----------------+----------------+----------------+----------------+
Joint Probability Distribution Formula:
P(e, i, m, a, s) = P(e) * P(i) * P(m | e, i) * P(a | m) * P(s | i)

Where:
 P(e): Probability of Exam Level
 P(i): Probability of IQ
 P(m | e, i): Probability of Marks given Exam Level and IQ
 P(a | m): Probability of Admission given Marks
 P(s | i): Probability of Aptitude Score given IQ

Enter the states for the following variables (leave blank for unknown):
Exam Level (e) [0=easy/1=difficult]: 0
IQ (i) [0=low/1=high]: 1
Marks (m) [0=low/1=high]: 0
Admission (a) [0=no/1=yes]:
Aptitude Score (s) [0=poor/1=good]: 1

Calculating JPD for the following combinations:
P(e=0, i=1, m=0, a=0, s=1) = 0.0050
P(e=0, i=1, m=0, a=1, s=1) = 0.0034

Total Joint Probability for the given states = 0.0084
PS C:\Users\niran\Downloads> 
```


