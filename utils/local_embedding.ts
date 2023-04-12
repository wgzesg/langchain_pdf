import { Embeddings } from 'langchain/embeddings';
import axios from 'axios';
import { AsyncCallerParams } from 'langchain/dist/util/async_caller';

export default class LocalEmbedding extends Embeddings{
    client: any;
    constructor(
        params: AsyncCallerParams
    ) {
        super(params);
        this.client = axios.create({
            baseURL: 'http://localhost:8000',
            timeout: 10000,
        });
    }

    async embedDocuments(texts: string[]): Promise<number[][]> {
        const embed: number[][] = [[0.1]];
        // sleep 1 second
        await new Promise((resolve) => setTimeout(resolve, 1000));
        return embed;
    }

    async embedQuery(text: string): Promise<number[]> {
        const response = await axios.post('http://127.0.0.1:8000/embed', {
            query: text,
          });
          console.log(response.data);
        return response.data;
    }

}