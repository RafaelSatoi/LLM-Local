from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from llama_index.embeddings.ollama import OllamaEmbedding as OllamaEmbeddingLlamma
from langchain_ollama import OllamaEmbeddings as OllamaEmbeddingsLangChain
from langchain_core.documents import Document as LCDocument

# Inicializing
print("Iniciando")

# Embedding Model Llama
embedding_name = "mxbai-embed-large:latest"
embedding = OllamaEmbeddingLlamma(model_name=embedding_name)

# LLM
llm = OllamaLLM(model="qwen3:4b-instruct")

choose = int(input("\nEscolha uma opção: \n1-Rag \n2-Banco Vetorizado\n\nNumero: "))

if choose == 1:

    # Change the Embedding Model
    print("\nTroca de Tipo Modelo Embedding")
    embedding = OllamaEmbeddingsLangChain(model=embedding_name)

    # Conection
    print("Conexão com a Base Local")
    db = FAISS.load_local("./banco_vetores/v2", embedding, allow_dangerous_deserialization=True)

    # Object
    retriever = db.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs={
            "k": 4,
            "score_threshold": 0.5
            }
        )

    # Prompt
    template_mensagens = ChatPromptTemplate.from_messages(
        [
            ("system", "Você está analisando um documento. Use as informações abaixo para responder a pergunta."),
            ("user", "Contexto:\n{context}\n\nPergunta:\n{question}")
        ]
    )

    # Question
    question = input("Pergunta: ")

    # Context
    print("--Contexto")
    relevant_docs = retriever.invoke(question)
    context =  "\n\n".join([doc.page_content for doc in relevant_docs])

    ## Show the chunks
    for i, doc in enumerate(relevant_docs, start=1):
        page = doc.metadata.get("page", "desconhecida")
        print(f"\n=== Trecho {i} (página {page}) ===\n")
        print(doc.page_content.strip()[:800])
        print("\n" + "-"*80)

    # Parser
    parser = StrOutputParser()


    # Chain
    print("--Chain")
    chain = template_mensagens | llm | parser
    result = chain.invoke({"context": context, "question": question})

    # Result
    print(result) 
    
elif choose == 2:

    # Loading PDF
    print("Carrega PDF")
    loader = PDFPlumberLoader("Historia.pdf")
    doc_loader = loader.load()

    # Modifying the text 
    doc_loader_ajuste = [
        Document(
            text=d.page_content,
            metadata=d.metadata 
        ) 
        for i, d in enumerate(doc_loader)
        if i < 2
    ]

    # Dynamic Parser
    print("Parser Dinâmico")
    splitter = SemanticSplitterNodeParser(
        embed_model=embedding,
        breakpoint_percentile_threshold=85,
        buffer_size=1
        )

    # Dynamic Chunks
    print("Chunks Dinâmicos")
    d_chunks = splitter.get_nodes_from_documents(doc_loader_ajuste)

    # Convert Doc Llama to Doc Langchain
    lc_docs = [
        LCDocument(page_content=node.text, metadata=node.metadata)
        for node in d_chunks
    ]

    # Change the Embedding Model
    print("Troca de Tipo Modelo Embedding")
    embedding = OllamaEmbeddingsLangChain(model=embedding_name)

    # Create DB Vector
    print("Criando Banco de Dados Vetorizados")

    db = FAISS.from_documents(lc_docs, embedding)
    db.save_local("./banco_vetores/v2")

    print("Sucesso ao Criar!")

else:
    print("Escolha um número de 1 a 2")