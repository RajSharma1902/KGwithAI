import os
import time
from dotenv import load_dotenv
from mistralai import Mistral
from neo4j import GraphDatabase
from pyvis.network import Network
from rapidfuzz import fuzz,process

load_dotenv()

# Initialize the Mistral client
client = Mistral(api_key=os.getenv("MISTRAL_API"))
model = "mistral-large-latest"
with open('/Users/rajnarayansharma/Desktop/Raj_924/RajNarayan_924/prompts.txt', 'r') as file:
    answerGen = file.read()

# Load the CSV-like dataset as a graph
def load_and_create_graph(file_path):
    graph = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 5:
                entity1, head_label, relation, entity2, tail_label = parts
                if entity1 not in graph:
                    graph[entity1] = []
                graph[entity1].append((relation, entity2))
                if entity2 not in graph:
                    graph[entity2] = []
                graph[entity2].append((f"Reverse of {relation}", entity1))
    return graph

# Retrieve relevant data from the graph based on the user query

# Retrieve relevant data from the graph based on the user query
def retrieve_relevant_chunks(query, graph):
    relevant_chunks = []
    all_entities = list(graph.keys())
    
    # Use fuzzy matching to find the best match for entities in the query
    for word in query.split():
        best_match = process.extractOne(word, all_entities, scorer=fuzz.ratio)
        if best_match and best_match[1] > 50:  # Threshold for matching accuracy
            matched_entity = best_match[0]
            for relation, related_entity in graph[matched_entity]:
                relevant_chunks.append(f"{matched_entity} {relation} {related_entity}")
    
    return relevant_chunks


# Build the prompt using relevant context
def build_prompt(relevant_context):
    prompt = (
        "You are an expert in analyzing csv datasets related to agriculture. "
        "The format of the data is (entity1, head_label, relation, entity2, tail_label). "
        "Here is the relevant data based on the question:\n"
    )
    
    for item in relevant_context:
        prompt += item + "\n"

    prompt += "\n" + answerGen  # Append answer generation instructions
    
    return prompt

# Neo4j connection function
def execute_cypher_query(cypher_query):
    # Connect to the Neo4j database
    uri = "bolt://localhost:7687"  # Change this to your Neo4j connection details if necessary
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))  # Username and password

    # Execute the Cypher query
    with driver.session() as session:
        result = session.run(cypher_query)
        data = result.data()  # Retrieve results as a list of dictionaries
    return data

# Create and save graph image using pyvis
def save_interactive_graph_with_pyvis(data):
    # Create a new pyvis network
    net = Network(height='500px', width='100%', notebook=True)

    # Add nodes and edges to the network
    for record in data:
        node1 = record['n']['name']
        node2 = record['m']['name']
        relation = record['r'][1]  # The relation is the second item in the tuple
        
        # Add nodes to the network if not already present
        net.add_node(node1)
        net.add_node(node2)
        
        # Add edge with relation label
        net.add_edge(node1, node2, label=relation)

    # Save the network as an interactive HTML file
    output_dir = "/Users/rajnarayansharma/Desktop/Raj_924/RajNarayan_924/Accuracy"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique file name for each graph
    graph_filename = os.path.join(output_dir, f"graph_{len(os.listdir(output_dir)) + 1}.html")
    
    # Generate and save the interactive graph
    net.show(graph_filename)
    print(f"Interactive graph saved as {graph_filename}")

# Load the graph once (outside the loop)
graph = load_and_create_graph('/Users/rajnarayansharma/Desktop/Raj_924/RajNarayan_924/fullAgri.txt')

# Question-answer loop
while True:
    question = input("What is your question? (Type 'exit' to quit): ")
    if question.lower() == 'exit':
        print("Exiting the program.")
        break

    # Retrieve relevant data from the graph based on the user's question (RAG retrieval)
    relevant_context = retrieve_relevant_chunks(question, graph)
    print("Relevant Chunks Extracted : ",relevant_context)
    # Build the prompt using the relevant context from the graph
    prompt = build_prompt(relevant_context)
    prompt += f"\nNow, if I ask the question: {question}\n"

    try:
        # Make the API call to Mistral
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=1,
            max_tokens=4096,
            top_p=1,
            stream=False
        )

        # Get and print the answer
        answer = chat_response.choices[0].message.content
        print(f"Answer: {answer}")
        
        # Extract Cypher query from the response
        cypher_query_start = "MATCH"
        cypher_query_end_variants = ["n,r,m", "n, r, m"]

        # Check if any of the variants of cypher_query_end are present in answer
        if cypher_query_start in answer:
            for cypher_query_end in cypher_query_end_variants:
                if cypher_query_end in answer:
                    start_idx = answer.index(cypher_query_start)
                    end_idx = answer.index(cypher_query_end) + len(cypher_query_end)
                    cypher_query = answer[start_idx:end_idx].strip()
                    print(f"Cypher Query: {cypher_query}")
                    break  # Stop checking once we find a match
            
            # Execute the Cypher query on Neo4j and get the result
            result_data = execute_cypher_query(cypher_query)
            print(f"Neo4j Data: {result_data}")
            
            # Save the graph image based on the result data
            save_interactive_graph_with_pyvis(result_data)
        else:
            print("No Cypher query found in the response.")

    except Exception as e:
        if "Rate limit reached" in str(e):
            print("Rate limit exceeded. Waiting to retry...")
            time.sleep(40)  # Wait for 40 seconds before retrying
        else:
            print(f"An error occurred: {e}")
