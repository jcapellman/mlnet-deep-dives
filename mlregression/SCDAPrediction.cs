using System;
using System.IO;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Common;
using mldeepdivelib.Enums;
using mldeepdivelib.Helpers;

using mlregression.Structures;

using Microsoft.ML;
using Microsoft.ML.Transforms.Normalizers;

namespace mlregression
{
    public class SCDAPrediction : BaseMLPrediction
    {
        protected override void Train(string[] args)
        {
            var modelObject = Activator.CreateInstance<EmploymentHistory>();

            var textReader = MlContext.Data.CreateTextLoader(columns: modelObject.ToColumns(), hasHeader: false, separatorChar: ',');

            var baseTrainingDataView = textReader.Read(args[(int)CommandLineArguments.INPUT_FILE]);

            var (name, min, max) = modelObject.GetLabelAttributes();

            var (trainData, testData) = MlContext.BinaryClassification.TrainTestSplit(baseTrainingDataView, testFraction: 0.1);

            var trainingDataView = MlContext.Data.FilterByColumn(trainData, name, min, max);
            var testDataView = MlContext.Data.FilterByColumn(testData, name, min, max);

            var dataProcessPipeline = MlContext.Transforms.CopyColumns("Label", name)
                .Append(MlContext.Transforms.Categorical.OneHotEncoding("PositionNameEncoded", "PositionName"))
                .Append(MlContext.Transforms.Normalize("IsMarried", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("BSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("MSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("YearsExperience", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("AgeAtHire", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("HasKids", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("WithinMonthOfVesting", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("DeskDecorations", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("LongCommute", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Concatenate("Features", "PositionNameEncoded", "IsMarried", "BSDegree", "MSDegree", "YearsExperience", "AgeAtHire", "HasKids", "WithinMonthOfVesting", "DeskDecorations", "LongCommute"));

            var trainer = MlContext.Regression.Trainers.StochasticDualCoordinateAscent();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingDataView);

            var dataWithPredictions = trainedModel.Transform(testDataView);

            var metrics = MlContext.BinaryClassification.Evaluate(dataWithPredictions, label: nameof(EmploymentHistory.DurationInMonths));

            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"AUC: {metrics.Auc}");
            Console.WriteLine($"F1 Score: {metrics.F1Score}");

            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall}");

            using (var fs = File.Create(args[(int)CommandLineArguments.OUTPUT_FILE]))
            {
                trainedModel.SaveTo(MlContext, fs);
            }

            Console.WriteLine($"Saved model to {args[(int)CommandLineArguments.OUTPUT_FILE]}");
        }

        protected override void Predict(string[] args)
        {
            var prediction = Predictor.Predict<EmploymentHistory, EmploymentHistoryPrediction>(MlContext, args[(int)CommandLineArguments.INPUT_FILE], args[(int)CommandLineArguments.OUTPUT_FILE]);

            Console.WriteLine($"Predicted Duration (in months): {prediction.DurationInMonths:0.#}");
        }
    }
}