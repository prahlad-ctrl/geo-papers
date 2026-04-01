import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE

def kar_graph():
    G = nx.karate_club_graph()
    return G

def node2vec_walk(G: nx.Graph, start_node: int, walk_length: int, p: float, q: float)->list[str]:
    walk = [start_node]
    for _ in range(walk_length-1):
        current = walk[-1]
        neighbors = list(G.neighbors(current))
        
        if (len(neighbors)) == 0: # no nodes neighbors, here they teleport in more adv models like pagerank but this breaks it so early stopping
            break
        
        if len(walk) == 1:
            next_node = random.choice(neighbors)
        else:
            prev = walk[-2]
            weights = []
            for n in neighbors:
                if n == prev:
                    weights.append(1/p)
                elif G.has_edge(prev, n):
                    weights.append(1.0)
                else:
                    weights.append(1/q)
        
            total = sum(weights)
            probs = [w/total for w in weights]
    
            next_node = np.random.choice(neighbors, p = probs, size = 1)[0]
            
        walk.append(next_node)
    
    return [str(n) for n in walk] # change id to str for gensim to import them in word2vec as words

def generate_walks(G: nx.Graph, num_walks: int, walk_length: int, p: float, q: float)->list[str]:
    nodes = list(G.nodes()) 
    walks = [] # it will cover all nodes and shuffling (num_walks x num_nodes[34])
    
    for walk_idx in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            w = node2vec_walk(G, node, walk_length, p, q)
            walks.append(w)
            
        if (walk_idx+1)%2 == 0 or walk_idx == 0:
            print(f"total walks so far: {len(walks)}")
    
    return walks
    
def train_node2vec(walks: list[list[str]], embed_dim = 64, window = 5, workers = 4, epochs = 5)->Word2Vec:
    model = Word2Vec(sentences = walks, # we basically did graph nodes into sentence based then passing it through word2vec
                     vector_size = embed_dim,
                     window=window,
                     sg=1, hs=1,
                     negative = 0,
                     min_count = 1,
                     workers=workers,
                     epochs=epochs,
                     seed = 42)
    
    return model

def extract_embed(G: nx.Graph, model: Word2Vec)->tuple[np.ndarray, list[str], list[int]]:
    node_ids = sorted(G.nodes())
    embed = np.array([model.wv[str(n)] for n in node_ids]) # dim(34 x 64)
    labels = [G.nodes[n]['club'] for n in node_ids]
    
    return embed, labels, node_ids

def similarity(G: nx.Graph, model: Word2Vec)->None:
    for node, score in model.wv.most_similar("0", topn=5): # this is for node 0, we can alter the ids to check any node
        club = G.nodes[int(node)]['club']
        print(f"For node 0, node {node:>2s}, cosine={score:.4f}, club={club}")
        
def visualise(embed: np.ndarray, labels: list[str]):
    coords = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(embed)
    unique_labels = list(set(labels))
    palette = ["#e63946", "#457b9d"]
    colour_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    colours = [colour_map[l] for l in labels]

    plt.figure(figsize=(7, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c=colours, s=80)
    plt.title("DeepWalk embeddings (t-SNE)")
 
    for lbl, col in colour_map.items():
        plt.scatter([], [], c=col, label=lbl)
    plt.legend()
 
    plt.savefig('img', dpi=150, bbox_inches="tight")    
        
def main():
    random.seed(42)
    np.random.seed(42)
    
    G = kar_graph()
    walks = generate_walks(G, num_walks=10, walk_length=40, p=1.0, q=0.5) # when p = q = 1, it acts as normal deepwalk
    model = train_node2vec(walks, embed_dim=64, window=5, workers=5, epochs=5)
    embed, labels, node_ids = extract_embed(G, model)
    similarity(G, model)
    visualise(embed, labels)
    
if __name__== '__main__':
    main()
