using System;
using System.IO;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Enums;
using mldeepdivelib.Helpers;

using ml_random_file_classification.Structures;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace ml_random_file_classification
{
    public class random_file_classification : BaseMLPrediction
    {
        protected override void Train(string[] args)
        {
            var data = MlContext.Data.ReadFromBinary(path: args[(int)CommandLineArguments.INPUT_FILE]);

            var pipeline = MlContext.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, nameof(FileData.Strings));

            var trainer = MlContext.BinaryClassification.Trainers.FastTree();
            var trainingPipeline = pipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(data);

            using (var fs = File.Create(args[(int)CommandLineArguments.OUTPUT_FILE]))
            {
                trainedModel.SaveTo(MlContext, fs);
            }

            Console.WriteLine($"Saved model to {args[(int)CommandLineArguments.OUTPUT_FILE]}");
        }

        protected override void Predict(string[] args)
        {
            var predictionData = new FileData
            {
                Strings = File.ReadAllBytes(args[(int)CommandLineArguments.INPUT_FILE]).ToString()
            };

            var prediction = Predictor.Predict<FileData, FilePrediction>(MlContext, args[(int)CommandLineArguments.OUTPUT_FILE], predictionData);

            var verdict = prediction.Prediction ? "Positive" : "Negative";

            Console.WriteLine($"{args[(int)CommandLineArguments.INPUT_FILE]} is predicted to be {verdict} | {prediction.Probability}");
        }
    }
}