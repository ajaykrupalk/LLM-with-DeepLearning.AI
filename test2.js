import dotenv from "dotenv";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
dotenv.config();

export async function loadAndSplitChunks({
    chunkSize,
    chunkOverlap
}) {
    const loader = new PDFLoader("./files/drylab.pdf");

    const rawCS229Docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap,
    });

    const splitDocs = await splitter.splitDocuments(rawCS229Docs);

    return splitDocs;
}

const splitDocs = await loadAndSplitChunks({
    chunkSize: 1536,
    chunkOverlap: 128,
});

function createContextFromSplitDocs(splitDocs) {
    const contextString = splitDocs.map(doc => doc.pageContent).join('\n\n'); // Combine with line breaks
    return contextString;
}

const prompt = ChatPromptTemplate.fromTemplate(`
Generate 3 questions about the context given below which a user can ask the LLM again.

<context>
{context}
<context>

`)

const res = RunnableSequence.from([
    RunnablePassthrough.assign({
        context: () => createContextFromSplitDocs(splitDocs)
    }),
    prompt,
    new ChatGoogleGenerativeAI({
        modelName: "gemini-pro",
        maxOutputTokens: 2048,
        apiKey: process.env.GOOGLE_API_KEY
    }),
    new StringOutputParser()
])

// const inputs = [
//     { question: 'Summarize the context given below.' },
//     { question: 'Generate 3 questions about the context given below which a user can ask the LLM again.' }
// ]
// const result = await res.batch(inputs)
const result = await res.invoke()

console.log(result)
