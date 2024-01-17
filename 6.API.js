import dotenv from "dotenv";
import express from "express";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HttpResponseOutputParser } from "langchain/output_parsers";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import * as parse from "pdf-parse";
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

export async function initializeVectorstoreWithDocuments({
    documents
}) {
    const embeddings = new GoogleGenerativeAIEmbeddings({
        modelName: "embedding-001", // 768 dimensions
        taskType: TaskType.RETRIEVAL_DOCUMENT,
        title: "Retrieval Document",
    });;
    const vectorstore = new MemoryVectorStore(embeddings);
    await vectorstore.addDocuments(documents);
    return vectorstore;
}

export function createDocumentRetrievalChain() {
    const convertDocsToString = (documents) => {
        return documents.map((document) => `<doc>\n${document.pageContent}\n</doc>`).join("\n");
    };

    const documentRetrievalChain = RunnableSequence.from([
        (input) => input.standalone_question,
        retriever,
        convertDocsToString,
    ]);

    return documentRetrievalChain;
}

export function createRephraseQuestionChain() {
    const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.`;

    const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
        ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
        new MessagesPlaceholder("history"),
        ["human", "Rephrase the following question as a standalone question:\n{question}"],
    ]);
    const rephraseQuestionChain = RunnableSequence.from([
        rephraseQuestionChainPrompt,
        new ChatGoogleGenerativeAI({
            modelName: "gemini-pro",
            maxOutputTokens: 2048,
            apiKey: process.env.GOOGLE_API_KEY
        }),
        new StringOutputParser(),
    ]);
    return rephraseQuestionChain;
}

export async function streamToString(stream) {
    const chunks = [];
    for await (const chunk of stream) {
        chunks.push(chunk);
    }
    return Buffer.concat(chunks).toString('utf-8');
}

// load the pdf file and split the document
const splitDocs = await loadAndSplitChunks({
    chunkSize: 1536,
    chunkOverlap: 128,
});

const vectorstore = await initializeVectorstoreWithDocuments({
    documents: splitDocs,
});

// // retrieve the document from the vectorstore
const retriever = vectorstore.asRetriever();

const documentRetrievalChain = createDocumentRetrievalChain();
const rephraseQuestionChain = createRephraseQuestionChain();

// make a prompt template
const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are an experienced researcher,
expert at interpreting and answering questions based on provided sources.
Using the below provided context and chat history, 
answer the user's question to the best of your ability
using only the resources provided. Be verbose!

<context>
{context}
</context>`;

const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
        "human",
        `Now, answer this question using the previous context and chat history:
  
    {standalone_question}`
    ]
]);

// retrieve the answer for the question
const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({
        standalone_question: rephraseQuestionChain,
    }),
    RunnablePassthrough.assign({
        context: documentRetrievalChain,
    }),
    answerGenerationChainPrompt,
    new ChatGoogleGenerativeAI({
        modelName: "gemini-pro",
        maxOutputTokens: 2048,
        apiKey: process.env.GOOGLE_API_KEY
    }),
]);

// "text/event-stream" is also supported
const httpResponseOutputParser = new HttpResponseOutputParser({
    contentType: "text/plain"
});

// With history of the chat response, retrieve the response
const messageHistory = new ChatMessageHistory();
// we should create a new history object per session
const messageHistories = {};
const getMessageHistoryForSession = (sessionId) => {
    if (messageHistories[sessionId] !== undefined) {
        return messageHistories[sessionId];
    }
    const newChatSessionHistory = new ChatMessageHistory();
    messageHistories[sessionId] = newChatSessionHistory;
    return newChatSessionHistory;
};
const finalRetrievalChain = new RunnableWithMessageHistory({
    runnable: conversationalRetrievalChain,
    getMessageHistory: getMessageHistoryForSession,
    inputMessagesKey: "question",
    historyMessagesKey: "history",
}).pipe(httpResponseOutputParser);

// Serving the API
const app = express();
const port = 3000;
app.use(express.json())
app.post('/', async (req, res) => {
    try {
        const stream = await finalRetrievalChain.stream({
            question: req.body.question
        }, { configurable: { sessionId: req.body.sessionId } });

        const resultString = await streamToString(stream);

        return res.status(200).type('text/plain').send(resultString)
    } catch (err) {
        console.error(err);
        return res.status(500).send('Error processing request: ' + err.message)
    }
});
app.listen(port, () => {
    console.log(`Listening on port ${port}`);
})

const decoder = new TextDecoder();

// readChunks() reads from the provided reader and yields the results into an async iterable
function readChunks(reader) {
  return {
    async* [Symbol.asyncIterator]() {
      let readResult = await reader.read();
      while (!readResult.done) {
        yield decoder.decode(readResult.value);
        readResult = await reader.read();
      }
    },
  };
}

const sleep = async () => {
  return new Promise((resolve) => setTimeout(resolve, 500));
}

let response, reader;

response = await fetch(`http://localhost:${port}`, {
    method: "POST",
    headers: {
        "content-type": "application/json",
    },
    body: JSON.stringify({
        question: "What are the prerequisites for this course?",
        sessionId: "1", // Should randomly generate/assign
    })
});

// response.body is a ReadableStream
reader = response.body?.getReader();

for await (const chunk of readChunks(reader)) {
  console.log("CHUNK:", chunk);
}

await sleep();

response = await fetch(`http://localhost:${port}`, {
  method: "POST",
  headers: {
    "content-type": "application/json",
  },
  body: JSON.stringify({
    question: "Can you list them in bullet point format?",
    sessionId: "1", // Should randomly generate/assign
  })
});

// response.body is a ReadableStream
reader = response.body?.getReader();

for await (const chunk of readChunks(reader)) {
  console.log("CHUNK:", chunk);
}

await sleep();

response = await fetch(`http://localhost:${port}`, {
  method: "POST",
  headers: {
    "content-type": "application/json",
  },
  body: JSON.stringify({
    question: "What did I just ask you?",
    sessionId: "2", // Should randomly generate/assign
  })
});

// response.body is a ReadableStream
reader = response.body?.getReader();

for await (const chunk of readChunks(reader)) {
  console.log("CHUNK:", chunk);
}

await sleep();