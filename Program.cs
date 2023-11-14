using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Plugins.Memory;
using SemanticMemoryAITextEmbedding;

/* You can build your own semantic memory combining a custom Embedding Generator
 * with a Memory storage that supports search by similarity (ie semantic search).
 *
 * In this example we use a volatile memory, a local simulation of a vector DB.
 *
 * You can replace VolatileMemoryStore with Qdrant (see QdrantMemoryStore connector)
 * or implement your connectors for Pinecone, Vespa, Postgres + pgvector, SQLite VSS, etc.
 */
const string memoryCollectionName = "SKGitHub";
var memoryWithCustomDb = new MemoryBuilder()
    .WithTextEmbeddingGeneration(new CustomTextEmbeddingGeneration())
    .WithMemoryStore(new VolatileMemoryStore())
    .Build();

await RunExampleAsync(memoryWithCustomDb);
static async Task RunExampleAsync(ISemanticTextMemory memory)
{
    await StoreMemoryAsync(memory);

    await SearchMemoryAsync(memory, "How do I get started?");

    /*
    Output:

    Query: How do I get started?

    Result 1:
      URL:     : https://github.com/microsoft/semantic-kernel/blob/main/README.md
      Title    : README: Installation, getting started, and how to contribute

    Result 2:
      URL:     : https://github.com/microsoft/semantic-kernel/blob/main/samples/dotnet-jupyter-notebooks/00-getting-started.ipynb
      Title    : Jupyter notebook describing how to get started with the Semantic Kernel

    */

    await SearchMemoryAsync(memory, "Can I build a chat with SK?");

    /*
    Output:

    Query: Can I build a chat with SK?

    Result 1:
      URL:     : https://github.com/microsoft/semantic-kernel/tree/main/samples/plugins/ChatPlugin/ChatGPT
      Title    : Sample demonstrating how to create a chat plugin interfacing with ChatGPT

    Result 2:
      URL:     : https://github.com/microsoft/semantic-kernel/blob/main/samples/apps/chat-summary-webapp-react/README.md
      Title    : README: README associated with a sample chat summary react-based webapp

    */
}
static async Task SearchMemoryAsync(ISemanticTextMemory memory, string query)
{
    Console.WriteLine("\nQuery: " + query + "\n");

    var memoryResults = memory.SearchAsync(memoryCollectionName, query, limit: 2, minRelevanceScore: 0.5);

    int i = 0;
    await foreach (MemoryQueryResult memoryResult in memoryResults)
    {
        Console.WriteLine($"Result {++i}:");
        Console.WriteLine("  URL:     : " + memoryResult.Metadata.Id);
        Console.WriteLine("  Title    : " + memoryResult.Metadata.Description);
        Console.WriteLine("  Relevance: " + memoryResult.Relevance);
        Console.WriteLine();
    }

    Console.WriteLine("----------------------");
}

static async Task StoreMemoryAsync(ISemanticTextMemory memory)
{
    /* Store some data in the semantic memory.
     *
     * When using Azure Cognitive Search the data is automatically indexed on write.
     *
     * When using the combination of VolatileStore and Embedding generation, SK takes
     * care of creating and storing the index
     */

    Console.WriteLine("\nAdding some GitHub file URLs and their descriptions to the semantic memory.");
    var githubFiles = SampleData();
    var i = 0;
    foreach (var entry in githubFiles)
    {
        await memory.SaveReferenceAsync(
            collection: memoryCollectionName,
            externalSourceName: "GitHub",
            externalId: entry.Key,
            description: entry.Value,
            text: entry.Value);

        Console.Write($" #{++i} saved.");
    }

    Console.WriteLine("\n----------------------");
}

static Dictionary<string, string> SampleData()
{
    return new Dictionary<string, string>
    {
        ["https://github.com/microsoft/semantic-kernel/blob/main/README.md"]
            = "README: Installation, getting started, and how to contribute",
        ["https://github.com/microsoft/semantic-kernel/blob/main/dotnet/notebooks/02-running-prompts-from-file.ipynb"]
            = "Jupyter notebook describing how to pass prompts from a file to a semantic plugin or function",
        ["https://github.com/microsoft/semantic-kernel/blob/main/dotnet/notebooks//00-getting-started.ipynb"]
            = "Jupyter notebook describing how to get started with the Semantic Kernel",
        ["https://github.com/microsoft/semantic-kernel/tree/main/samples/plugins/ChatPlugin/ChatGPT"]
            = "Sample demonstrating how to create a chat plugin interfacing with ChatGPT",
        ["https://github.com/microsoft/semantic-kernel/blob/main/dotnet/src/SemanticKernel/Memory/VolatileMemoryStore.cs"]
            = "C# class that defines a volatile embedding store",
    };
}