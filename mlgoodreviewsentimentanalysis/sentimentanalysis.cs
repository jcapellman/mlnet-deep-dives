using System;
using System.IO;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Enums;
using mldeepdivelib.Helpers;

using mlgoodreviewsentimentanalysis.Structures;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace mlgoodreviewsentimentanalysis
{
    public class sentimentanalysis : BaseMLPrediction
    {
        protected override void Train(string[] args)
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<SentimentData>(args[(int)CommandLineArguments.INPUT_FILE]);
            
            var pipeline = MlContext.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, nameof(SentimentData.Text));

            var trainer = MlContext.BinaryClassification.Trainers.FastTree();
            var trainingPipeline = pipeline.Append(trainer);
            
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            using (var fs = File.Create(args[(int)CommandLineArguments.OUTPUT_FILE]))
            {
                trainedModel.SaveTo(MlContext, fs);
            }

            Console.WriteLine($"Saved model to {args[(int)CommandLineArguments.OUTPUT_FILE]}");
        }

        protected override void Predict(string[] args)
        {
            var predictionData = new SentimentData
            {
                Text = args[(int) CommandLineArguments.OUTPUT_FILE]
            };

            var prediction = Predictor.Predict<SentimentData, SentimentPrediction>(MlContext, args[(int)CommandLineArguments.INPUT_FILE], predictionData);

            var verdict = prediction.Prediction ? "Positive" : "Negative";

            Console.WriteLine($"{predictionData.Text} is predicted to be {verdict} | {prediction.Probability}");
        }
    }
}