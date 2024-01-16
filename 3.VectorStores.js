import dotenv from "dotenv";
dotenv.config();

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { similarity } from "ml-distance";
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

let res;
// Embeddings are vector representations of text
// They capture the semantic meaning of the text
const embeddings = new GoogleGenerativeAIEmbeddings({
    modelName: "embedding-001", // 768 dimensions
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    title: "Retrieval Document",
});

// res = await embeddings.embedQuery("OK Google");
// console.log(res, res.length);

// Checking if the vectors are similar
// const vector1 = await embeddings.embedQuery("What are vectors useful for in machine learning?");
// const vector2 = await embeddings.embedQuery("A group of parrots is called a pandemonium.");
// res = similarity.cosine(vector1, vector2); 
// console.log(res)//Output: "0.5919134776580153"

// const vector1 = await embeddings.embedQuery("What are vectors useful for in machine learning?");
// const vector2 = await embeddings.embedQuery("Vectors are representations of information.");
// res = similarity.cosine(vector1, vector2);
// console.log(res)//Output: "0.8827628756538672" - Higher than the above similarity

/**
 * Loading a PDF file
 * Saving the file to an in-memory vector store
 * Getting 4 very similar text
 */
const loader = new PDFLoader("./files/compressed.pdf");
const PDFdoc = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 128,
    chunkOverlap: 0,
});
const splitDocs = await splitter.splitDocuments(PDFdoc);

const vectorstore = new MemoryVectorStore(embeddings);
await vectorstore.addDocuments(splitDocs); //adding the split documents into the in-memory vector store

// searching for similar documents
// const retrieveDocs = await vectorstore.similaritySearch(
//     "What is deep learning",
//     4
// )
// Formatting the response
// const pageContents = retrieveDocs.map(doc => doc.pageContent)

// console.log(pageContents);

// Retrievers
const retriever = vectorstore.asRetriever();
res = await retriever.invoke("What is deep learning?")
console.log(res);