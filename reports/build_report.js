// Builds reports/Freight_Rate_Report.docx from the figures and numbers in results/.
//   node reports/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, LevelFormat, BorderStyle,
} = require("docx");

const FIG = path.join(__dirname, "figures");
const png = (name, width) => {
  const buf = fs.readFileSync(path.join(FIG, name));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  return new ImageRun({ type: "png", data: buf, transformation: { width, height: Math.round((width * h) / w) },
    altText: { title: name, description: name, name } });
};

const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120 }, ...opts,
  children: Array.isArray(text) ? text : [new TextRun(text)] });
const b = (text) => new TextRun({ text, bold: true });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (parts) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 },
  children: Array.isArray(parts) ? parts : [new TextRun(parts)] });
const caption = (t) => p([new TextRun({ text: t, italics: true, size: 18, color: "52514E" })], { spacing: { after: 200 } });
const img = (name, width) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [png(name, width)] });

const border = { style: BorderStyle.SINGLE, size: 4, color: "D0CEC6" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(widths, header, rows, boldRow = -1, allLeft = false) {
  const total = widths.reduce((a, c) => a + c, 0);
  const cell = (text, i, head, bold) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "E8EEF4", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: i === 0 || allLeft ? AlignmentType.LEFT : AlignmentType.RIGHT,
      children: [new TextRun({ text, bold: head || bold, size: 19 })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true, true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((t, i) => cell(t, i, false, ri === boldRow)) }))],
  });
}

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Freight Rate Prediction: Approach and Validation")] }),
  p([new TextRun({ text: "Machine Learning Engineer assessment. Code, data checks and every number below are reproducible from the repository (see section 9).", color: "52514E" })]),

  h1("1. Summary"),
  p("The task is to predict the posted rate for 12,000 loads in November and December 2025 from 48,000 labelled loads dated January to October. The validation set is entirely in the future, so I validated by time rather than at random."),
  p("The final model is a gradient-boosted tree on static load attributes (distance, weight, equipment, pickup and delivery coordinates), trained on rate per mile after setting aside 1.4% of labels that look corrupt, followed by a one-parameter linear correction for how far that day's market_index sits from its month's average. On four forward-in-time folds (July to October) it reaches a median absolute percentage error of 1.75% and a mean absolute error of $101 per load, against 3.20% and $135 for a per-mile lookup baseline."),
  p("The most useful finding was about market_index: used naively it made out-of-time predictions worse, even though it looked informative. Section 5 explains what is actually going on."),

  h1("2. Data findings and how they were handled"),
  table([2300, 3500, 3560], ["Issue", "What I found", "What I did"], [
    ["Negative weights", "292 train and 145 validation rows. |weight| runs 5,000 to 47,500 and the negatives mirror that range.", "Treated as sign errors: used the absolute value."],
    ["Missing values", "weight: 300 train, 165 validation. market_index: 374 train, 249 validation (about 0.8%).", "weight filled with the training median. market_index recovered from the other loads on the same date."],
    ["Weight ceiling", "2.5% of loads sit at exactly 47,500 in both files.", "Left as is. It looks like a cap and the model sees it equally in both periods."],
    ["Corrupt-looking rates", "682 train rows (1.42%) are 4+ robust sigma from a robust fit: 341 about 2 to 5 times too high, 341 about 0.2 to 0.3 times too low. Spread evenly over months and equipment.", "Excluded from fitting inside each fold. Kept in every scoring metric, since the evaluator's labels will contain them too."],
    ["Odd distances", "111 rows (0.23%) have a recorded distance far above the straight-line distance from the coordinates.", "Tested as a feature (road/straight-line ratio); it changed nothing, so I dropped it."],
    ["Unseen cities", "8 validation cities never appear in training (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego). 12.1% of loads touch one.", "No city identity in the model. Geography enters only through coordinates, which tested better on held-out cities."],
    ["Unused columns", "quote_signal has 0.05 correlation with rate per mile and is mostly row-to-row noise.", "Tested twice and left out (section 5)."],
  ], -1, true),
  p(""),
  p("The December chart file contains no market_index or quote_signal. That constraint shaped the design: a model that needs those columns could not produce the required chart without inventing them."),

  h1("3. Train/validation split"),
  img("split.png", 560),
  caption("Figure 1. Expanding-window validation. Each fold trains on every month up to k and scores month k+1. Folds 1 to 4 score July to October."),
  p("The validation file starts on 1 November, after the last training date, so the real task is forecasting the next two months. A random split would let the model train on days that surround each test day and would reward memorising a particular day. Section 5 shows this is not hypothetical here: the same feature is judged in opposite ways by a random split and by a forward-in-time split."),
  p("Features are computed without labels, and the date-level market_index average only looks at rows with the same date. Corrupt-label trimming is fitted on each fold's training rows only. tests/test_sanity.py checks that test dates are strictly later than training dates and that the features are identical with the label column removed."),

  h1("4. Model"),
  bullet([b("Target: "), new TextRun("log of rate per mile. Rate scales with distance, so predicting the per-mile rate removes most of the variance and keeps errors proportional. Predictions are converted back to dollars by multiplying by distance.")]),
  bullet([b("Stage 1: "), new TextRun("scikit-learn HistGradientBoostingRegressor on log distance, |weight|, equipment and the four coordinates. It never sees the date, so it cannot memorise individual days.")]),
  bullet([b("Stage 2: "), new TextRun("a Huber-loss linear regression of stage 1's out-of-fold residuals on the day's market_index minus its month average. Using out-of-fold residuals keeps the correction honest; in-sample residuals are too small.")]),
  bullet([b("Why trees at all: "), new TextRun("the lookup baseline and a robust linear model both lose clearly (table 1). Regional effects in the coordinates need a nonlinear model. Hyperparameters barely matter: all 12 settings I tried land between 2.10% and 2.11% clean MAPE.")]),

  h1("5. Experiments"),
  p([b("Scoring convention. "), new TextRun("All numbers are the mean of the four forward-in-time folds. \"All rows\" includes the corrupt-looking labels, which no model can predict. \"Clean rows\" excludes them and is how I compare models, because the corrupt rows add the same unavoidable noise to every candidate. The clean-row flag comes from a robust fit to all training data, so it is a diagnostic for reporting, not something used to train.")]),
  h2("5.1 Does the model beat sensible baselines? (table 1)"),
  table([4560, 1200, 1200, 1200, 1200], ["Model", "MAE $", "Median APE", "Clean MAE $", "Clean MAPE"], [
    ["Median rate per mile by equipment and distance band", "134.8", "3.20%", "81.2", "3.71%"],
    ["Robust linear (distance, weight, equipment)", "118.9", "2.58%", "65.1", "2.94%"],
    ["GBM, raw labels", "108.2", "2.03%", "54.3", "2.30%"],
    ["GBM, absolute-error loss, raw labels", "105.2", "1.90%", "51.3", "2.15%"],
    ["GBM, corrupt labels set aside", "104.2", "1.86%", "50.3", "2.11%"],
    ["Final: GBM + month-deviation market adjustment", "101.0", "1.75%", "47.0", "1.99%"],
  ], 5),
  caption("Table 1. Temporal cross-validation, mean of four folds. Setting aside corrupt labels beat both raw training and an absolute-error loss that was meant to be robust."),
  h2("5.2 How should market_index be used? (table 2)"),
  p("The static model's out-of-fold daily residual correlates 0.70 with that day's market_index (304 days), so the signal is real. But handing it to the model as an ordinary input made predictions worse."),
  table([5160, 1050, 1050, 1050, 1050], ["Variant", "MAE $", "Median APE", "Clean MAE $", "Clean MAPE"], [
    ["No market input", "104.2", "1.86%", "50.3", "2.11%"],
    ["Joint GBM + absolute market_index", "129.1", "3.04%", "75.5", "3.17%"],
    ["Joint GBM + deviation from month average", "107.7", "1.98%", "53.8", "2.27%"],
    ["Two-stage, absolute market_index", "126.4", "2.98%", "72.7", "3.11%"],
    ["Two-stage, deviation from month average (final)", "101.0", "1.75%", "47.0", "1.99%"],
    ["Two-stage, deviation from 31-day centred average", "100.1", "1.69%", "46.1", "1.95%"],
    ["Two-stage, deviation + day of week", "102.8", "1.79%", "48.7", "2.07%"],
    ["Two-stage, deviation + quote_signal", "101.5", "1.82%", "47.6", "2.05%"],
  ], 4),
  caption("Table 2. Ways of using market_index and quote_signal."),
  img("market_index_by_month.png", 520),
  caption("Figure 2. Out-of-fold daily rate residual against market_index. In six of the ten months the two move together with correlation above 0.95, but each month sits at its own level. In January and September the slopes are similar and the levels differ by about 6 points."),
  p("That is why the absolute index fails: it implies a level that does not carry over to a new month. The day's deviation from its own month's average is the part that does transfer, and a one-parameter correction on it improves the four-fold average. Adding day of week or quote_signal did not help. A 31-day centred reference was slightly better (1.95% vs 1.99% clean MAPE); I kept the calendar-month version because it is simpler to explain and the difference is small."),
  p("The split itself also matters. With a random 5-fold split, adding the absolute market_index looks like a large improvement (clean MAPE 2.92% to 2.13%); with the forward-in-time split it is a large loss (2.11% to 3.17%). A random split would have led me to ship the wrong model."),
  h2("5.3 Other decisions"),
  bullet([b("Static inputs: "), new TextRun("removing weight costs 0.85 points of clean MAPE, removing coordinates 0.74, removing equipment 2.9. All three stay.")]),
  bullet([b("City identity: "), new TextRun("no gain on seen cities (2.11% vs 2.12%) and worse on held-out cities (3.43% vs 2.97%), so the model uses coordinates only.")]),
  bullet([b("Forecast scaling: "), new TextRun("multiplying predictions by 0.98 to 1.05 never beat 1.00 by more than $0.3 MAE, so I left the forecast uncalibrated.")]),
  bullet([b("Unseen cities cost little: "), new TextRun("holding out whole cities gives 2.97% clean MAPE, against 2.92% for a random split over the same months, so coordinates generalise to new cities about as well as to known ones. The two numbers are not comparable with the forward-in-time folds, which cover the easier July to October months.")]),

  h1("6. Fixed December prediction"),
  img("candidate_december.png", 590),
  caption("Figure 3. Chart produced by the provided score.py from the final model (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb)."),
  p("Predictions range from $809 to $833, mean $821. The lane's historical Dry Van median was roughly $780 to $880 by month (21 loads), so the level is consistent with history."),
  p([b("How to read the shape. "), new TextRun("Only the date varies, so every wiggle comes from stage 2. The chart file has no market_index, so I used each date's value from validation.csv, which is a feature column and contains no labels. The weekly cycle (Thursdays highest, Sundays lowest) reflects the day-of-week pattern in that index. The curve does not model Christmas or New Year: the training data has no holiday evidence, and the 25 December peak is just a Thursday.")]),
  p([b("Assumption to flag. "), new TextRun("If the intent was a pure forecast with no knowledge of December's index, the honest curve is nearly flat at about $821. The adjustment moves the predictions by at most about 1.5% around that level.")]),

  h1("7. Error analysis"),
  p("On the October fold (trained January to September), median error is 1.48% over all rows and 1.44% on clean rows. Mean error is 4.7% overall because 80 corrupt-looking labels average 182% error each. Observations:"),
  bullet("Short hauls under 300 miles are the weakest segment: 7.0% mean error versus 2 to 5% in the other distance bands. A fixed dollar error is a larger share of a small rate."),
  bullet("Reefer loads have the highest mean error (5.4% vs 4.4% for Dry Van and Flatbed), but the median errors are similar across equipment types, so this is mostly the corrupt labels."),
  bullet("Only 8 October loads were on a lane unseen in training, too few to say anything about unseen lanes from this fold. The city-holdout experiment (5.3) covers that question instead."),

  h1("8. Limitations and what I would do next"),
  bullet([b("Monthly level is unpredictable from this data. "), new TextRun("Month-average residuals have a standard deviation of about 3 points (from -5% in January to +5% in June) and market_index does not explain that. With one year of data I cannot tell whether November and December carry a seasonal offset; a prior year would settle it. This is the biggest risk to the validation score.")]),
  bullet([b("Corrupt labels. "), new TextRun("About 1.4% of validation labels will be unpredictable noise. If the unseen metric is RMSE or MAE they dominate the total, so differences between good models may look small.")]),
  bullet([b("Model selection used the same four folds that are reported. "), new TextRun("The decisions that matter are large (2.11% vs 3.17%), but the small differences, such as 1.95% vs 1.99%, should not be over-read.")]),
  bullet([b("Next steps: "), new TextRun("a prior year to estimate seasonality; an explanation for the corrupt labels from the data owner; prediction intervals; monitoring the stage-2 slope for drift.")]),

  h1("9. Reproduction"),
  p("python -m pip install -r requirements.txt, place the provided CSVs in data/, then run src/data_checks.py, src/experiments.py (about 10 minutes), src/train_predict.py and score.py as described in README.md. Everything is seeded and deterministic."),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 40, bold: true, color: "0B0B0B", font: "Calibri" }, paragraph: { spacing: { after: 120 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, color: "064A56", font: "Calibri" }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 23, bold: true, color: "0B0B0B", font: "Calibri" }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1200, right: 1440, bottom: 1200, left: 1440 } } }, children }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(path.join(__dirname, "Freight_Rate_Report.docx"), buf); console.log("wrote report"); });
