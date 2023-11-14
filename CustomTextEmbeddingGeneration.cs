using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using Microsoft.SemanticKernel.AI.Embeddings;
using Microsoft.SemanticKernel.AI.Embeddings.VectorOperations;

namespace SemanticMemoryAITextEmbedding;

// inherit from ITextEmbeddingGeneration: this is the interface that the semantic kernel will call to generate embeddings
public class CustomTextEmbeddingGeneration : ITextEmbeddingGeneration
{
    // semantic kernel will call this method to generate embeddings
    public Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(IList<string> data, CancellationToken cancellationToken = default)
    {
        // Create a new ML context
        var mlContext = new MLContext();
        // Load the data
        var textData = LoadData(data);

        // Create and train the model
        var textTransformer = CreateAndTrainModel(mlContext);
        // Embed sentences
        var transformedTexts = EmbedSentences(textData, textTransformer, mlContext);

        IList<ReadOnlyMemory<float>> results = transformedTexts.Select(text => new ReadOnlyMemory<float>(text.Features)).ToList();

        return Task.FromResult(results);
    }

    // load the data into a list of TextData objects it is what the ML.NET pipeline expects
    private List<TextData> LoadData(IList<string> data)
    {
        return data.Select(item => new TextData { Text = item }).ToList();
    }

    // we are not training a model, but we still need to create a model to get the transformer
    private ITransformer CreateAndTrainModel(MLContext mlContext)
    {
        // define the data preparation pipeline
        var dataPrepPipeline = mlContext.Transforms.Text.NormalizeText("Text")
            .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))
            .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Features", "Tokens", WordEmbeddingEstimator.PretrainedModelKind.GloVe50D));

        // even though we are not 'training' in the traditional sense, we still call 'Fit' to set up the transformer
        ITransformer dataPrepTransformer = dataPrepPipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<TextData>()));

        return dataPrepTransformer;
    }

    // embed the sentences
    private List<TransformedTextData> EmbedSentences(IEnumerable<TextData> data, ITransformer textTransformer, MLContext mlContext)
    {
        // load data into IDataView
        var dataView = mlContext.Data.LoadFromEnumerable(data);

        // transform the data
        var transformedData = textTransformer.Transform(dataView);

        // convert transformed data to an enumerable
        return mlContext.Data.CreateEnumerable<TransformedTextData>(transformedData, reuseRowObject: false).ToList();
    }

    // we can use this to find the closest match to a sentence
    private string FindClosestSentence(string sentenceToMatch, IEnumerable<TransformedTextData> transformedTexts, ITransformer textTransformer, MLContext mlContext)
    {
        // Transform the sentence to match
        var sentenceToMatchData = new[] { new TextData { Text = sentenceToMatch } };
        var sentenceToMatchDataView = mlContext.Data.LoadFromEnumerable(sentenceToMatchData);
        var transformedSentenceToMatch = textTransformer.Transform(sentenceToMatchDataView);
        var sentenceToMatchFeatures = mlContext.Data.CreateEnumerable<TransformedTextData>(transformedSentenceToMatch, reuseRowObject: false).First().Features;

        return transformedTexts.Select(x => new { Sentence = x.Text, Distance = CalculateDistance(x.Features, sentenceToMatchFeatures) })
            .OrderBy(x => x.Distance)
            .First().Sentence;
    }

    // calculate the distance between two vectors
    private double CalculateDistance(IReadOnlyList<float> vectorA, IReadOnlyList<float> vectorB)
    {
        // might be better to use internal vector operations
        var distance = vectorA.ToArray().CosineSimilarity(vectorB.ToArray());
        
        // euclidean distance calculation is what ml.net example uses
        //var distance = vectorA.Select((t, i) => Math.Pow(t - vectorB[i], 2)).Sum();

        // return the square root of the distance
        return Math.Sqrt(distance);
    }
}

// this is the class that the ML.NET pipeline expects
internal class TransformedTextData : TextData
{
    public float[] Features { get; set; }
}

// this is the class that the ML.NET pipeline expects
internal class TextData
{
    public string Text { get; set; }
}
