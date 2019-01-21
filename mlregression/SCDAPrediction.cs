using System;
using System.IO;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Common;
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

            var textReader = MlContext.Data.CreateTextReader(columns: modelObject.ToColumns(), hasHeader: false, separatorChar: ',');

            var baseTrainingDataView = textReader.Read(args[1]);

            var label = modelObject.GetLabelAttributes();

            var trainingDataView = MlContext.Data.FilterByColumn(baseTrainingDataView, label.Name, label.Min, label.Max);

            var dataProcessPipeline = MlContext.Transforms.CopyColumns(label.Name, "Label")
                .Append(MlContext.Transforms.Categorical.OneHotEncoding("PositionName", "PositionNameEncoded"))
                .Append(MlContext.Transforms.Normalize("IsMarried", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize("BSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize(inputName: "MSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize(inputName: "YearsExperience", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize(inputName: "AgeAtHire", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize(inputName: "HasKids", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize(inputName: "WithinMonthOfVesting", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize(inputName: "DeskDecorations", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Normalize(inputName: "LongCommute", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Concatenate("Features", "PositionNameEncoded", "IsMarried", "BSDegree", "MSDegree", "YearsExperience", "AgeAtHire", "HasKids", "WithinMonthOfVesting", "DeskDecorations", "LongCommute"));

            var trainer = MlContext.Regression.Trainers.StochasticDualCoordinateAscent();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingDataView);

            using (var fs = File.Create(args[2]))
            {
                trainedModel.SaveTo(MlContext, fs);
            }

            Console.WriteLine($"Saved model to {args[2]}");
        }

        protected override void Predict(string[] args)
        {
            var prediction = Predictor.Predict<EmploymentHistory, EmploymentHistoryPrediction>(MlContext, args[1], args[2]);

            Console.WriteLine($"Predicted Duration (in months): {prediction.DurationInMonths:0.#}");
        }

        protected override void FeatureExtraction(string[] args)
        {
            throw new NotImplementedException();
        }
    }
}