import dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";

dotenv.config();

let res;

// initialize the Gemini-Pro Model
const model = new ChatGoogleGenerativeAI({
    modelName: "gemini-pro",
    maxOutputTokens: 2048,
    apiKey: process.env.GOOGLE_API_KEY
})

// Invoke the model to generate the response
// res = await model.invoke([
//     [
//         "human",
//         "Tell me a joke"
//     ]
// ])

// A prompt template refers to a reproducible way to generate a prompt
const prompt = ChatPromptTemplate.fromTemplate("What are three good names for a company that makes {product}?")

// provide the value for the dynamic variable
// Ouput: "Human: What are three good names for a company that makes open source software?"
// res = await prompt.format({
//     product: "open source software"
// })

// Ouputs in LLM format
// res = await prompt.formatMessages({
//     product: "open source software"
// })

// A better way to output in LLM format
// const promptFromMessage = ChatPromptTemplate.fromMessages([
//     ["system","You are an expert at picking company names"],
//     ["human","What are three good names for a company that makes {product}?"]
// ])

// res = await promptFromMessage.formatMessages({
//     product: "open source software"
// })

// LCEL: LangChain Expression Language
// This helps generate texts, answer questions or analyze data
// const chain = prompt.pipe(model);
// res = await chain.invoke({
//     product: 'colorful socks'
// })

// The StringOutputParser is responsible for converting the structured format to string format
const outputParser = new StringOutputParser();
// const nameGenerationChain = prompt.pipe(model).pipe(outputParser)
// res = await nameGenerationChain.invoke({
//     product: 'open source  software'
// })

// Runnables: a composition of multiple runnables that are executed sequentially in a predefined order
// The runnables below are prompt, model, outputParser
const nameGenerationChain = RunnableSequence.from([
    prompt,
    model,
    outputParser
])
// res = await nameGenerationChain.invoke({
//     product: 'fancy cookies'
// })

// Streaming: To give the output chunk by chunk
// This is an advantage since LLM outputs take time to process
// const stream = await nameGenerationChain.stream({
//     product: "really cool robots",
// });

// for await (const chunk of stream) {
//     console.log(chunk);
// }

// Batch() used to process multiple inputs efficiently in a batch or parallelized manner
const inputs = [
    { product: 'large calculators' },
    { product: 'alpaca wool sweaters' }
]
res = await nameGenerationChain.batch(inputs)

console.log(res);