using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

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
            var data = MlContext.Data.ReadFromTextFile<FileData>(path: args[(int)CommandLineArguments.INPUT_FILE]);

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
                Strings = File.ReadAllBytes(args[(int) CommandLineArguments.INPUT_FILE]).ToString()
            };

            var prediction = Predictor.Predict<FileData, FilePrediction>(MlContext,
                args[(int) CommandLineArguments.OUTPUT_FILE], predictionData);

            var verdict = prediction.Prediction ? "Positive" : "Negative";

            Console.WriteLine(
                $"{args[(int) CommandLineArguments.INPUT_FILE]} is predicted to be {verdict} | {prediction.Probability}");
        }

        protected override void FeatureExtraction(string[] args)
        {
            var startDate = DateTime.Now;

            var files = Directory.GetFiles(args[(int)CommandLineArguments.INPUT_FILE]);

            var sb = new StringBuilder();

            foreach (var filePath in files)
            {
                var extraction = FeatureExtractFile(filePath);

                if (extraction == null)
                {
                    continue;
                }

                sb.AppendLine(extraction.ToString());
            }

            File.WriteAllText(args[(int)CommandLineArguments.OUTPUT_FILE], sb.ToString());

            Console.WriteLine($"Feature Extraction completed in {DateTime.Now.Subtract(startDate).TotalMinutes} minutes to {args[(int)CommandLineArguments.OUTPUT_FILE]}");
        }

        private FileData FeatureExtractFile(string filePath)
        {
            var fileData = new FileData();

            Memory<byte> data = File.ReadAllBytes(filePath);

            var regex = new Regex("\\w{4,}", RegexOptions.Compiled);

            fileData.Strings = string.Join(",", regex.Matches(data.ToString()).Select(a => a.Value));

            return fileData;
        }
    }
}