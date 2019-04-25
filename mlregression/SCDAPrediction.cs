using System;
using System.IO;
using System.Linq;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Enums;
using mldeepdivelib.Helpers;

using mlregression.Structures;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace mlregression
{
    public class SCDAPrediction : BaseMLPrediction
    {
        protected override void Train(string[] args)
        {
            var modelObject = Activator.CreateInstance<EmploymentHistory>();

            IDataView baseTrainingDataView = MlContext.Data.LoadFromTextFile<EmploymentHistory>(args[(int)CommandLineArguments.INPUT_FILE], hasHeader: true, separatorChar: ',');
            var testDataView = MlContext.Data.LoadFromTextFile<EmploymentHistory>(args[(int)CommandLineArguments.INPUT_FILE], hasHeader: true, separatorChar: ',');
            var cnt = baseTrainingDataView.GetColumn<float>(nameof(EmploymentHistory.DurationInMonths)).Count();
            IDataView trainingDataView = MlContext.Data.FilterRowsByColumn(baseTrainingDataView, nameof(EmploymentHistory.DurationInMonths), lowerBound: 1, upperBound: 150);
            var cnt2 = trainingDataView.GetColumn<float>(nameof(EmploymentHistory.DurationInMonths)).Count();

            var dataProcessPipeline = MlContext.Transforms.CopyColumns("Label", nameof(EmploymentHistory.DurationInMonths))
                .Append(MlContext.Transforms.Categorical.OneHotEncoding("PositionNameEncoded", "PositionName"))
                .Append(MlContext.Transforms.NormalizeMeanVariance("IsMarried"))
                .Append(MlContext.Transforms.NormalizeMeanVariance("BSDegree"))
                .Append(MlContext.Transforms.NormalizeMeanVariance("MSDegree"))
                .Append(MlContext.Transforms.NormalizeMeanVariance("YearsExperience")
                .Append(MlContext.Transforms.NormalizeMeanVariance("AgeAtHire"))
                .Append(MlContext.Transforms.NormalizeMeanVariance("HasKids"))
                .Append(MlContext.Transforms.NormalizeMeanVariance("WithinMonthOfVesting"))
                .Append(MlContext.Transforms.NormalizeMeanVariance("DeskDecorations"))
                .Append(MlContext.Transforms.NormalizeMeanVariance("LongCommute"))
                .Append(MlContext.Transforms.Concatenate("Features", "PositionNameEncoded", "IsMarried", "BSDegree", "MSDegree", "YearsExperience", "AgeAtHire", "HasKids", "WithinMonthOfVesting", "DeskDecorations", "LongCommute")));

            var trainer = MlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingDataView);

            var dataWithPredictions = trainedModel.Transform(testDataView);

            var metrics = MlContext.BinaryClassification.Evaluate(dataWithPredictions, predictedLabelColumnName: nameof(EmploymentHistoryPrediction.DurationInMonths));

            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve}");
            Console.WriteLine($"F1 Score: {metrics.F1Score}");

            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall}");

            using (var fs = File.Create(args[(int)CommandLineArguments.OUTPUT_FILE]))
            {
                MlContext.Model.Save(trainedModel, trainingDataView.Schema, fs);
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