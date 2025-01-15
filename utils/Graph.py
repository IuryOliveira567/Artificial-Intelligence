from collections import deque
from heapq import heapify, heappop, heappush

class Vertex(object):

    def __init__(self, vertex_id):

        self.id = vertex_id
        self.edges = []
    
class Edge(object):

    def __init__(self, from_vertex, to_vertex, cost=1):

        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.cost = cost

        assert cost >= 0, (f"Cost cannot be negative: {self}, cost={cost}")

    def __repr__(self):

         return f"{self.from_vertex} --> {self.to_vertex}"
         
class Graph(object):

    def __init__(self):

        self.vertices = {}

    def add_vertex(self, vertex):
        
        if(isinstance(vertex, Vertex) and vertex.id not in self.vertices):
            self.vertices[vertex.id] = vertex
            return True
        else:
            return False

    def add_edge(self, edge):

        v1 = edge.from_vertex
        v2 = edge.to_vertex
        
        if(isinstance(edge, Edge)):    
          if(v1 in self.vertices and v2 in self.vertices):
              self.vertices[v1].edges.append(edge)

              rEdge = Edge(v2, v1, edge.cost)
              self.vertices[v2].edges.append(rEdge)
              return True
          else:
              return False

    def get_vertices(self):

        return self.vertices.keys()

    def __iter__(self):

        return iter(self.vertices.values())

    def dfs(self, start_vertex_id, visited=set()):

        start_vertex = self.vertices.get(start_vertex_id)

        if not start_vertex:
           return []

        dfs_order = []
        
        visited.add(start_vertex_id)
        dfs_order.append(start_vertex_id)

        for edge in start_vertex.edges:
            if edge.to_vertex not in visited:
               dfs_order.extend(self.dfs(edge.to_vertex, visited.copy()))

        return dfs_order

    def bfs(self, start_vertex_id):

        queue = deque()
        visited = set()
        
        queue.append(start_vertex_id)
        visited.add(start_vertex_id)
            
        while(queue):
            
            current_vertex = queue.popleft()
            
            for edge in self.vertices[current_vertex].edges:
                if(edge.to_vertex not in visited):                   
                   queue.append(edge.to_vertex)
                   visited.add(edge.to_vertex)

        return visited

    def dijkstra(self, start_vertex, end_vertex):

        distances = {node: float("inf") for node in self.vertices}
        distances[start_vertex] = 0

        pq = [(0, start_vertex)]
        heapify(pq)

        visited = set()

        while(pq):
            current_distance, current_vertex = heappop(pq)

            if(current_vertex in visited):
                continue

            visited.add(current_vertex)

            for edge in self.vertices[current_vertex].edges:
                try_distance = current_distance + edge.cost

                if(try_distance < distances[edge.to_vertex]):
                    distances[edge.to_vertex] = try_distance
                    heappush(pq, (try_distance, edge.to_vertex))
                
        return distances[end_vertex]

    def heuristic(self, vertex, goal):

        return 1

    def a_star_search(self, start_vertex_id, goal_vertex_id):

        open_set = set([start_vertex_id])
        came_from = {}
        g_score = {start_vertex_id: 0}
        f_score = {start_vertex_id: self.heuristic(start_vertex_id, goal_vertex_id)}

        while open_set:
            current_id = min(open_set, key=lambda vertex_id: f_score[vertex_id])
            current = self.vertices[current_id]
            
            if(current.id == goal_vertex_id):
                return self.reconstruct_path(came_from, current_id)

            open_set.remove(current_id)

            for neighbor_edge in current.edges:
                neighbor = neighbor_edge.to_vertex
                
                try_g_score = g_score[current_id] + neighbor_edge.cost

                if(self.vertices[neighbor].id not in g_score or try_g_score < g_score[self.vertices[neighbor].id]):
                    came_from[self.vertices[neighbor].id] = current.id
                    g_score[self.vertices[neighbor].id] = try_g_score
                    f_score[self.vertices[neighbor].id] = g_score[self.vertices[neighbor].id] + self.heuristic(self.vertices[neighbor].id, goal_vertex_id)

                    if(self.vertices[neighbor].id not in open_set):
                        open_set.add(self.vertices[neighbor].id)

        return None

    def reconstruct_path(self, came_from, current_id):

        total_path = [self.vertices[current_id]]

        while(current_id in came_from):
            current_id = came_from[current_id]
            total_path.insert(0, self.vertices[current_id])

        return total_path
