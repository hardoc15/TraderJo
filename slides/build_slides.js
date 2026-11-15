// Builds slides/TraderJo_slides.pptx (3 slides, ~5 minutes).
// Usage: npm install pptxgenjs && node slides/build_slides.js
const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT = path.join(__dirname, "TraderJo_slides.pptx");
const THEME = {
  name: "TraderJo",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F24", lt1: "FFFFFF", dk2: "0D3B3F", lt2: "EEF3F2",
    accent1: "13896F", accent2: "E3A33B", accent3: "9AA3AC", accent4: "C2362F",
    accent5: "2A6F97", accent6: "6C5B7B", hlink: "13896F", folHlink: "6C5B7B",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "TraderJo: a reinforcement-learning paper trader";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---- layouts
const titleBox = { x: 0.6, y: 0.4, w: 12.1, h: 0.9 };
pres.defineSlideMaster({
  title: "DARK_TITLE_ONLY",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", ...titleBox, fontSize: 36, bold: true,
      color: C.background1, fontFace: THEME.headFontFace, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.4, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent3 },
});
pres.defineSlideMaster({
  title: "LIGHT_TITLE_ONLY",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", ...titleBox, fontSize: 36, bold: true,
      color: C.text2, fontFace: THEME.headFontFace, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.4, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent3 },
});

function numberedStep(slide, n, x, y, head, body, dark) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: C.accent2 },
    line: { color: C.accent2 }, objectName: `step${n}-circle` });
  slide.addText(String(n), { x, y, w: 0.5, h: 0.5, align: "center", valign: "middle", fontSize: 16,
    bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: `step${n}-num` });
  slide.addText([
    { text: head, options: { bold: true, fontSize: 16, color: dark ? C.background1 : C.text2, breakLine: true } },
    { text: body, options: { fontSize: 14, color: dark ? C.background2 : C.text1 } },
  ], { x: x + 0.7, y: y - 0.08, w: 4.9, h: 0.95, valign: "top", margin: 0, isTextBox: true,
       objectName: `step${n}-text` });
}

// ================================================================ slide 1
{
  pres.addSection({ title: "What we built" });
  const s = pres.addSlide({ masterName: "DARK_TITLE_ONLY", sectionTitle: "What we built" });
  s.addText("TraderJo: an RL agent that paper-trades ETFs", { placeholder: "title" });
  s.addText("Harrison Cohen · COMP 560 Final Project · github.com/hardoc15/TraderJo", {
    x: 0.6, y: 1.3, w: 12, h: 0.4, fontSize: 14, color: C.accent3, margin: 0, isTextBox: true,
    objectName: "byline" });

  const steps = [
    ["Observe", "32 daily price features (returns, volatility, RSI, trend) for SPY, QQQ, TLT, GLD, plus current holdings"],
    ["Act", "PPO policy network outputs target portfolio weights (softmax over 4 ETFs + cash)"],
    ["Reward", "Log portfolio return after 10 bps transaction costs"],
    ["Evaluate", "Train 2006–19 · validate 2020–21 · test 2022–26 (never seen); 6 variants × 5 seeds"],
  ];
  steps.forEach(([h, b], i) => numberedStep(s, i + 1, 0.6, 2.05 + i * 1.18, h, b, true));

  s.addImage({ path: path.join(__dirname, "dashboard.png"), x: 6.75, y: 1.95, w: 5.95, h: 4.86,
    altText: "Screenshot of the TraderJo dashboard showing a $100k paper account, current allocation, equity curve and trades",
    objectName: "dashboard-screenshot", shadow: { type: "outer", color: "000000", opacity: 0.35, blur: 8, offset: 3, angle: 90 } });
  s.addText("Local web dashboard: simulated $100k account, weekly rebalances", {
    x: 6.75, y: 6.88, w: 5.6, h: 0.35, fontSize: 11, italic: true, color: C.accent3, margin: 0,
    isTextBox: true, objectName: "screenshot-caption" });

  s.addNotes(
    "We built TraderJo, a reinforcement-learning agent that manages a paper portfolio of four ETFs: US stocks (SPY), " +
    "the Nasdaq (QQQ), long-term Treasuries (TLT) and gold (GLD), plus cash. Each step the agent observes 32 technical " +
    "features and its current holdings, outputs portfolio weights, and is rewarded with its log return after a 10 basis " +
    "point trading cost. We trained it with PPO in a custom Gymnasium environment. Data is split strictly by time: the " +
    "agent never sees 2022 onward during training or model selection. We trained six variants with five random seeds " +
    "each and compared them to standard portfolios. On the right is the product: a local dashboard that runs the deployed " +
    "agent on a simulated $100k account and can optionally mirror orders to an Alpaca paper-trading account.");
}

// ================================================================ slide 2
{
  pres.addSection({ title: "Results" });
  const s = pres.addSlide({ masterName: "LIGHT_TITLE_ONLY", sectionTitle: "Results" });
  s.addText("RL matched simple portfolios but didn't beat SPY", { placeholder: "title" });

  // test-period Sharpe ratios (results/summary.csv, results/static_mix.csv)
  const rows = [
    ["Buy & hold SPY", 0.70, false], ["Agent's mix, held fixed", 0.67, false],
    ["Equal weight (monthly)", 0.67, false], ["SMA-200 trend rule", 0.66, false],
    ["PPO weekly (deployed)", 0.62, true], ["PPO monthly", 0.60, true],
    ["60/40 stocks/bonds", 0.32, false], ["Momentum top-2", 0.28, false],
    ["PPO daily", 0.23, true],
  ].reverse(); // bar charts plot bottom-up
  s.addChart(pres.charts.BAR, [
    { name: "PPO agent (5-seed ensemble)", labels: rows.map(r => r[0]), values: rows.map(r => (r[2] ? r[1] : 0)) },
    { name: "Baseline", labels: rows.map(r => r[0]), values: rows.map(r => (r[2] ? 0 : r[1])) },
  ], {
    x: 0.6, y: 1.45, w: 7.4, h: 5.5, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 45,
    chartColors: [HEX.accent1, HEX.accent3],
    showTitle: true, title: "Sharpe ratio on held-out test period (2022–Nov 2026, after costs)",
    titleFontSize: 13, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
    showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: "0.00;-0.00;;",
    dataLabelColor: HEX.lt1, dataLabelFontSize: 12, dataLabelFontBold: true, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: "6B737C", valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0",
    valGridLine: { color: "E3E6EA", size: 0.75 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: HEX.dk1, legendFontFace: "+mn-lt",
    objectName: "sharpe-chart",
  });

  const stats = [
    ["19×", "less trading when the agent rebalances weekly instead of daily (24.6% → 1.3% of the portfolio per day)"],
    ["0.70 → 0.23", "daily PPO's Sharpe before vs. after 10 bps costs: the edge is eaten by trading"],
    ["−20.7%", "max drawdown of weekly PPO vs. −24.5% for SPY: smoother, but lower return"],
  ];
  stats.forEach(([big, small], i) => {
    const y = 1.6 + i * 1.8;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.45, y, w: 4.3, h: 1.55, rectRadius: 0.12,
      fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `stat${i + 1}-card` });
    s.addText([
      { text: big, options: { fontSize: 30, bold: true, color: C.accent1, fontFace: THEME.headFontFace, breakLine: true } },
      { text: small, options: { fontSize: 13, color: C.text1 } },
    ], { x: 8.7, y: y + 0.18, w: 3.85, h: 1.25, valign: "top", margin: 0, isTextBox: true,
         objectName: `stat${i + 1}-text` });
  });

  s.addNotes(
    "Here are the held-out test results, 2022 to November 2026, all after transaction costs. The green bars are our PPO " +
    "agents, each an ensemble of five seeds; grey bars are baselines. The weekly and monthly agents reach a Sharpe ratio " +
    "around 0.6, about the same as an equal-weight portfolio or a trend rule, and below simply buying SPY at 0.70. " +
    "The daily agent is much worse. Before costs it scores 0.70, but it turns over a quarter of the portfolio every day, " +
    "and the 10 basis point cost drops it to 0.23. Making the agent decide weekly cut trading by about 19 times. " +
    "The weekly agent does have a smaller worst-case drawdown than SPY, minus 21 versus minus 24 percent. " +
    "The random agent, not shown, loses 57 percent purely to trading costs.");
}

// ================================================================ slide 3
{
  pres.addSection({ title: "Takeaways" });
  const s = pres.addSlide({ masterName: "DARK_TITLE_ONLY", sectionTitle: "Takeaways" });
  s.addText("What we learned", { placeholder: "title" });

  const cards = [
    ["Trading costs decide the design", "How often the agent may trade mattered more than the reward function or risk penalty. Daily RL agents learned to over-trade."],
    ["It learned to diversify, not to time", "The agent settles on ~20% in each asset plus cash. Holding that mix fixed scores 0.67, better than the agent's own timing (0.62)."],
    ["Short validation windows mislead", "Picking by 2020–21 validation Sharpe chose daily PPO (1.49): the worst variant on the test period. Bull-market years reward risk."],
  ];
  cards.forEach(([head, body], i) => {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: 3.8, h: 3.25, rectRadius: 0.12,
      fill: { color: "16525A" }, line: { color: "16525A" }, objectName: `card${i + 1}` });
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 2.05, w: 0.55, h: 0.55, fill: { color: C.accent2 },
      line: { color: C.accent2 }, objectName: `card${i + 1}-circle` });
    s.addText(String(i + 1), { x: x + 0.3, y: 2.05, w: 0.55, h: 0.55, align: "center", valign: "middle",
      fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: `card${i + 1}-num` });
    s.addText(head, { x: x + 0.3, y: 2.8, w: 3.2, h: 0.8, valign: "top", margin: 0, fontSize: 19, bold: true,
      color: C.background1, fontFace: THEME.headFontFace, isTextBox: true, objectName: `card${i + 1}-head` });
    s.addText(body, { x: x + 0.3, y: 3.65, w: 3.2, h: 1.2, valign: "top", margin: 0, fontSize: 14,
      color: C.background2, isTextBox: true, objectName: `card${i + 1}-body` });
  });

  s.addText([
    { text: "Next steps  ", options: { bold: true, color: C.accent2 } },
    { text: "walk-forward validation · turnover-aware rewards · macro and sentiment signals · forward paper trading", options: { color: C.background1 } },
  ], { x: 0.6, y: 5.45, w: 12.1, h: 0.45, fontSize: 15, margin: 0, isTextBox: true, objectName: "next-steps" });
  s.addText("Code, results and dashboard: github.com/hardoc15/TraderJo  ·  Paper trading only, not financial advice", {
    x: 0.6, y: 6.1, w: 11.5, h: 0.4, fontSize: 12, color: C.accent3, margin: 0, isTextBox: true, objectName: "footer-link" });

  s.addNotes(
    "Three lessons. First, transaction costs dominate: the single most important design choice was how often the agent " +
    "is allowed to trade, more than the reward function or adding a risk penalty. Second, when we looked at what the agent " +
    "actually does, it holds a nearly fixed, diversified mix of about 20 percent per asset. If you simply hold its " +
    "average mix without any timing, you score slightly better, so the RL part learned diversification rather than " +
    "market timing. Third, model selection is fragile: we fixed a rule in advance to deploy the variant with the best " +
    "2020–21 validation Sharpe, and it picked the daily agent, which was the worst on the test period. We report that " +
    "honestly and deploy the weekly ensemble, whose real out-of-sample test is now forward paper trading. Next we would " +
    "use walk-forward validation over several regimes and add turnover-aware rewards. Thanks!");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  try {
    const { applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js");
    await applyTheme(OUT, THEME);
  } catch (e) {
    console.warn("theme colors not applied:", e.message);
  }
  console.log("wrote", OUT);
})();
