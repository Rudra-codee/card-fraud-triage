"""
build_paper_pdf.py

Generates a publication-quality IEEE-style 2-column conference paper PDF
for Phase 1 and Phase 2 of the Card Fraud Triage project using ReportLab.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, Image, FrameBreak, HRFlowable, NextPageTemplate
)
from reportlab.pdfgen import canvas

# Page geometry
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 40  # 40 pt margin
GUTTER = 16  # gutter between two columns
COL_WIDTH = (PAGE_WIDTH - 2 * MARGIN - GUTTER) / 2  # ~260 pt
CONTENT_HEIGHT = PAGE_HEIGHT - 2 * MARGIN

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor("#444444"))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            header_text = "HIWANJ et al.: DENSITY, RECONSTRUCTION, OR BOTH? A LEAK-FREE STUDY OF FRAUD ANOMALY TRIAGE"
            self.drawCentredString(PAGE_WIDTH / 2.0, PAGE_HEIGHT - 28, header_text)
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(MARGIN, PAGE_HEIGHT - 32, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 32)

        # Footer (all pages)
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(PAGE_WIDTH - MARGIN, 25, footer_text)
        conf_text = "Advanced Machine Learning & Deep Learning --- Capstone Project (Phases 1 and 2)"
        self.drawString(MARGIN, 25, conf_text)
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(MARGIN, 34, PAGE_WIDTH - MARGIN, 34)
        
        self.restoreState()


def create_paper_pdf(output_path="paper.pdf"):
    doc = BaseDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )

    # First page: Top title frame + 2 column frames below
    TITLE_HEIGHT = 156
    frame_title = Frame(
        MARGIN, PAGE_HEIGHT - MARGIN - TITLE_HEIGHT,
        PAGE_WIDTH - 2 * MARGIN, TITLE_HEIGHT,
        id='F_Title', topPadding=0, bottomPadding=2, leftPadding=0, rightPadding=0
    )
    frame_p1_col1 = Frame(
        MARGIN, MARGIN,
        COL_WIDTH, CONTENT_HEIGHT - TITLE_HEIGHT,
        id='F_P1_C1', topPadding=4, bottomPadding=4, leftPadding=0, rightPadding=4
    )
    frame_p1_col2 = Frame(
        MARGIN + COL_WIDTH + GUTTER, MARGIN,
        COL_WIDTH, CONTENT_HEIGHT - TITLE_HEIGHT,
        id='F_P1_C2', topPadding=4, bottomPadding=4, leftPadding=4, rightPadding=0
    )

    # Later pages: full height 2 column frames
    frame_col1 = Frame(
        MARGIN, MARGIN,
        COL_WIDTH, CONTENT_HEIGHT,
        id='F_Col1', topPadding=8, bottomPadding=4, leftPadding=0, rightPadding=4
    )
    frame_col2 = Frame(
        MARGIN + COL_WIDTH + GUTTER, MARGIN,
        COL_WIDTH, CONTENT_HEIGHT,
        id='F_Col2', topPadding=8, bottomPadding=4, leftPadding=4, rightPadding=0
    )

    template_page1 = PageTemplate(id='Page1', frames=[frame_title, frame_p1_col1, frame_p1_col2])
    template_later = PageTemplate(id='LaterPages', frames=[frame_col1, frame_col2])
    doc.addPageTemplates([template_page1, template_later])

    # Typography Styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=16.5,
        leading=20.5,
        alignment=1, # Center
        textColor=colors.HexColor("#111111"),
        spaceAfter=5
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10.5,
        leading=13.5,
        alignment=1,
        textColor=colors.HexColor("#222222"),
        spaceAfter=9
    )

    authors_style = ParagraphStyle(
        'DocAuthors',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9.2,
        leading=11.8,
        alignment=1,
        textColor=colors.HexColor("#111111"),
        spaceAfter=3
    )

    affil_style = ParagraphStyle(
        'DocAffil',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.2,
        leading=10.6,
        alignment=1,
        textColor=colors.HexColor("#333333"),
        spaceAfter=6
    )

    abstract_body = ParagraphStyle(
        'AbstractBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.2,
        leading=10.5,
        alignment=4, # Justify
        textColor=colors.HexColor("#1a1a1a")
    )

    index_terms = ParagraphStyle(
        'IndexTerms',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=8.2,
        leading=10.5,
        textColor=colors.black
    )

    sec_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9.0,
        leading=11.2,
        alignment=1, # Center
        spaceBefore=6,
        spaceAfter=2.5,
        textColor=colors.HexColor("#000000")
    )

    subsec_heading = ParagraphStyle(
        'SubSectionHeading',
        parent=styles['Normal'],
        fontName='Times-BoldItalic',
        fontSize=8.4,
        leading=10.4,
        spaceBefore=4,
        spaceAfter=2.0,
        textColor=colors.HexColor("#000000")
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.0,
        leading=10.1,
        alignment=4, # Justify
        spaceAfter=2.8,
        textColor=colors.HexColor("#111111")
    )

    table_title_style = ParagraphStyle(
        'TableTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=7.4,
        leading=8.8,
        alignment=1,
        spaceAfter=2.0
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=6.8,
        leading=8.0,
        alignment=1
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=6.8,
        leading=8.0,
        alignment=1
    )

    fig_caption = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.0,
        leading=8.6,
        alignment=1,
        spaceBefore=2.0,
        spaceAfter=3.5
    )

    ref_style = ParagraphStyle(
        'References',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.0,
        leading=8.4,
        alignment=4,
        leftIndent=10,
        firstLineIndent=-10,
        spaceAfter=1.8
    )

    story = []

    # ==================== TITLE FRAME ====================
    story.append(Paragraph("Density, Reconstruction, or Both?", title_style))
    story.append(Paragraph("A Leak-Free Study of Gaussian Mixture Models, Deep Autoencoders, and Two-Tier Triage for Credit Card Fraud", subtitle_style))
    story.append(Paragraph("Kaustubh Hiwanj, Rudraksh Rathod, Bhargav Patil, Anurag, Pratyush Chouksey", authors_style))
    story.append(Paragraph("Department of Computer Science and Engineering &mdash; Card Fraud Triage Research Group<br/>Advanced Machine Learning and Deep Learning &mdash; Capstone Project (Phases 1 and 2)", affil_style))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#666666"), spaceAfter=5, spaceBefore=0))

    # Move to Column 1 on Page 1
    story.append(FrameBreak())
    story.append(NextPageTemplate('LaterPages'))

    # ==================== ABSTRACT & INDEX TERMS ====================
    abs_text = (
        "<b><i>Abstract</i>&mdash;Credit card fraud detection is characterized by extreme class imbalance (0.17%) and strict operational review capacity constraints. Standard supervised classifiers suffer from severe alert fatigue and struggle against evolving zero-day fraud strategies. In this paper, we present a leak-free benchmark comparing classical probabilistic density estimation with deep neural reconstruction for unsupervised fraud anomaly triage. Every experimental decision is governed by a strict chronological protocol (Strategy B: 60/20/20 split) with leak-free preprocessing fitted exclusively on legitimate training transactions, and a single-pass held-out test evaluation. In Phase 1, we benchmark a frozen single-component Gaussian Mixture Model (GMM) baseline against a 12-candidate model-selection grid (<i>K</i> &isin; {2, 3, 4, 5}, full, diagonal, and tied covariance). While diagonal covariance fails completely (Average Precision &approx; 0.045&ndash;0.059), a 4-component full-covariance model achieves an Average Precision (PR-AUC) of 0.4840 on validation data (6.5&times; over the 0.0740 baseline). In Phase 2, we evaluate a 5-layer symmetric PyTorch Autoencoder (29 &rarr; 20 &rarr; 10 &rarr; 5 &rarr; 10 &rarr; 20 &rarr; 29) trained with legitimate-only validation early stopping, quantifying anomaly severity via Mean Squared Reconstruction Error (MSE). Finally, a one-time evaluation on the untouched held-out test set (<i>N</i> = 56,962, 75 frauds) benchmarks all frozen candidates against frozen operating thresholds. The <i>K</i> = 4 full-covariance GMM demonstrates dominant precision under constrained capacity (78.00% precision in the Top 50 alerts, intercepting 39 of 75 frauds), while the unimodal GMM baseline delivers superior broad-spectrum recall (84.00% recall at a frozen 5% false positive rate). We synthesize these findings into a Two-Tier Operational Triage Architecture that unifies rapid high-precision step-up challenges with a high-recall background safety net.</b>"
    )
    story.append(Paragraph(abs_text, abstract_body))
    story.append(Spacer(1, 3))
    
    terms_text = "<b><i>Index Terms</i>&mdash;Credit card fraud triage, anomaly detection, Gaussian mixture models, deep autoencoders, negative log-likelihood, reconstruction error, data leakage, precision-recall AUC, review capacity.</b>"
    story.append(Paragraph(terms_text, index_terms))
    story.append(Spacer(1, 5))

    # ==================== SECTION I: INTRODUCTION ====================
    story.append(Paragraph("I. INTRODUCTION", sec_heading))
    story.append(Paragraph(
        "Commercial payment security systems process hundreds of thousands of transactions daily, yet fraudulent transactions represent less than 0.2% of overall traffic [1]. Because human fraud investigation desks operate under finite daily review capacity, deploying models that generate uncalibrated false alarms causes severe alert fatigue, allowing financial loss to slip through uninspected. Furthermore, supervised models trained on past fraud signatures overfit to historical attack patterns and frequently fail against adversarial zero-day fraud [2].",
        body_style
    ))
    story.append(Paragraph(
        "To circumvent these vulnerabilities, this project implements an unsupervised <b>normal-behavior anomaly triage paradigm</b>: models are trained strictly on legitimate cardholder transactions (<i>y</i><sub>train</sub> == 0), learning the distribution and geometry of legitimate spending habits. Prospective transactions are then ranked by anomaly severity to maximize fraud interception under strict false positive budgets.",
        body_style
    ))
    story.append(Paragraph(
        "We investigate two distinct anomaly detection paradigms: (1) <i>Statistical Density Estimation (GMM)</i>, which quantifies statistical rarity via Negative Log-Likelihood (NLL = &minus;ln <i>p</i>(<i>x</i>)); and (2) <i>Deep Subspace Reconstruction (Autoencoder)</i>, which measures geometrical deviation via per-sample Mean Squared Reconstruction Error (MSE = <sup>1</sup>/<sub><i>D</i></sub>&sum;(<i>x<sub>d</sub></i> &minus; <i>x&#770;<sub>d</sub></i>)<sup>2</sup>).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Problem statement.</b> Given 284,807 credit card transactions across 48 continuous hours, we evaluate: (a) how far classical full-covariance Gaussian densities can rank fraud anomalies; (b) whether multi-modal mixtures resolve distinct spending behaviors; (c) whether deep non-linear reconstruction provides superior precision; and (d) how both paradigms behave under strict operational analyst review quotas&mdash;all under a single protocol in which every choice is made on validation data and the test set is touched exactly once.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Hypotheses.</b> Phase 1 closed with two falsifiable hypotheses: <b>H1</b>&mdash;a deep symmetric autoencoder trained strictly on legitimate data exceeds the single-component GMM baseline's Average Precision (AP > 0.0740) on validation data; and <b>H2</b>&mdash;non-linear subspace compression resolves subtle transaction anomalies without suffering from the Expectation-Maximization (EM) seed instability observed in multi-component full-covariance GMMs.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Contributions:</b><br/>"
        "1) <i>A Leak-Free Protocol:</i> Strict chronological partitioning (Strategy B: 60/20/20) with timestamp boundary alignment at index 170,888, eliminating future-to-past temporal leakage.<br/>"
        "2) <i>12-Model GMM Selection:</i> Systematic evaluation of component counts (<i>K</i> &isin; {2, 3, 4, 5}) and covariance structures (`full`, `diag`, `tied`), with AIC/BIC and condition audits.<br/>"
        "3) <i>Deep Autoencoder Benchmarking:</i> Symmetric 5-layer PyTorch network with legitimate-only validation early stopping and MSE scoring.<br/>"
        "4) <i>Single-Pass Test Benchmark:</i> One-time evaluation on 56,962 untouched transactions (75 frauds) with frozen validation thresholds.<br/>"
        "5) <i>Two-Tier Operational Architecture:</i> A deployment design combining rapid high-precision interception (78% precision at Top 50 alerts) with a broad safety net (84% recall at 5% FPR).",
        body_style
    ))

    # ==================== SECTION II: RELATED WORK ====================
    story.append(Paragraph("II. RELATED WORK AND POSITIONING", sec_heading))
    story.append(Paragraph(
        "Early benchmarks on the European card fraud dataset [1] focused on supervised classifiers (Random Forests, Logistic Regression, XGBoost). While supervised models achieve high ROC-AUC, they suffer from two operational weaknesses. First, ROC-AUC is misleadingly optimistic under extreme imbalance [3]: an FPR of 1% over 56,900 legitimate transactions generates 569 false alarms, overwhelming triage teams. Second, synthetic oversampling techniques (e.g., SMOTE [4]) distort the high-dimensional covariance structure of normal payments.",
        body_style
    ))
    story.append(Paragraph(
        "Unsupervised anomaly detection formulates fraud detection as deviation from normalcy [5]. Gaussian Mixture Models (GMM) model normal behavior as a mixture of Gaussian ellipsoids [6], allowing closed-form log-likelihood scoring. Conversely, Deep Autoencoders compress input vectors through a dimensional bottleneck; because the network optimizes reconstruction loss exclusively on legitimate data, anomalies incur elevated reconstruction errors [7]. Prior comparative studies frequently evaluated models on random splits or refitted scalers on test data. This study enforces complete test isolation and frozen threshold provenance.",
        body_style
    ))

    # ==================== SECTION III: PROBLEM FORMULATION ====================
    story.append(Paragraph("III. PROBLEM FORMULATION AND METRICS", sec_heading))
    story.append(Paragraph(
        "Let <i>D</i> = {(<i>x<sub>n</sub></i>, <i>y<sub>n</sub></i>)} where <i>x<sub>n</sub></i> &isin; <b>R</b><sup>29</sup> represents standardized transaction features and <i>y<sub>n</sub></i> &isin; {0, 1} denotes ground truth (0 = legitimate, 1 = fraud). Models are fitted strictly on legitimate training observations <i>D</i><sub>train, legit</sub> = {<i>x<sub>n</sub></i> | <i>y<sub>n</sub></i> = 0, <i>n</i> &isin; Train}.",
        body_style
    ))
    story.append(Paragraph(
        "<b>GMM Density Scoring:</b> A <i>K</i>-component GMM models probability density as <i>p</i>(<i>x</i> | &theta;) = &sum;<sub><i>k</i>=1</sub><sup><i>K</i></sup> &pi;<sub><i>k</i></sub> <i>N</i>(<i>x</i> | &mu;<sub><i>k</i></sub>, &Sigma;<sub><i>k</i></sub>). The anomaly score is defined as Negative Log-Likelihood (NLL):",
        body_style
    ))
    story.append(Paragraph(
        "&nbsp;&nbsp;&nbsp;&nbsp;<i>S</i><sub>GMM</sub>(<i>x</i>) = &minus;ln <i>p</i>(<i>x</i> | &theta;) = <sup>29</sup>/<sub>2</sub> ln(2&pi;) + <sup>1</sup>/<sub>2</sub> ln|&Sigma;| + <sup>1</sup>/<sub>2</sub> (<i>x</i> &minus; &mu;)<sup><i>T</i></sup> &Sigma;<sup>&minus;1</sup> (<i>x</i> &minus; &mu;)",
        body_style
    ))
    story.append(Paragraph(
        "Higher scores denote statistical rarity under normal transaction density.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Autoencoder Reconstruction Scoring:</b> The Autoencoder decomposes into an encoder <i>f</i><sub>&phi;</sub>: <b>R</b><sup>29</sup> &rarr; <b>R</b><sup>5</sup> and decoder <i>g</i><sub>&psi;</sub>: <b>R</b><sup>5</sup> &rarr; <b>R</b><sup>29</sup>. Anomaly severity is measured by per-sample Mean Squared Error (MSE):",
        body_style
    ))
    story.append(Paragraph(
        "&nbsp;&nbsp;&nbsp;&nbsp;<i>S</i><sub>AE</sub>(<i>x</i>) = <sup>1</sup>/<sub>29</sub> &sum;<sub><i>d</i>=1</sub><sup>29</sup> (<i>x<sub>d</sub></i> &minus; <i>x_rec<sub>d</sub></i>)<sup>2</sup>, &nbsp;&nbsp; <i>x_rec</i> = <i>g</i><sub>&psi;</sub>(<i>f</i><sub>&phi;</sub>(<i>x</i>))",
        body_style
    ))
    story.append(Paragraph(
        "<b>Operational Metrics:</b><br/>"
        "&bull; <i>Average Precision (AP / PR-AUC):</i> Area under the Precision-Recall curve, weighted by recall changes. Baseline random performance equals fraud prevalence (&approx; 0.0010).<br/>"
        "&bull; <i>Recall @ Fixed FPR (&le; 1% and &le; 5%):</i> Fraction of fraudulent transactions intercepted when the threshold is set to the lowest value ensuring legitimate false positive rate &le; &alpha;.<br/>"
        "&bull; <i>Top-N Review Capacity:</i> Fraud yield and precision when the top <i>N</i> &isin; {50, 100, 200, 500} alerts are audited.",
        body_style
    ))

    # ==================== SECTION IV: DATA & PROTOCOL ====================
    story.append(Paragraph("IV. DATA AND EXPERIMENTAL PROTOCOL", sec_heading))
    story.append(Paragraph(
        "<b>Chronological Strategy B Partitioning:</b> Random splitting shuffles future transactions into the training set, causing temporal leakage. We enforce strict chronological partitioning across 48 continuous hours (September 2013):<br/>"
        "&bull; <i>Train (60%):</i> 170,888 transactions (170,528 legitimate, 360 fraud), 0.0s&ndash;120,396.0s.<br/>"
        "&bull; <i>Validation (20%):</i> 56,957 transactions (56,900 legitimate, 57 fraud), 120,397.0s&ndash;145,247.0s.<br/>"
        "&bull; <i>Test (20%):</i> 56,962 transactions (56,887 legitimate, 75 fraud), 145,248.0s&ndash;172,792.0s.<br/>"
        "The nominal 60% boundary fell inside a 1-second burst sharing Time = 120,396.0s; we shifted the boundary by 4 rows to index 170,888 to ensure transaction atomicity.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Data Quality Decisions:</b> (i) <i>Zero-Dollar Transactions (1,825 rows):</i> Fully retained. Merchant authorization pings represent account verification checks; fraud prevalence is 5.49% (32&times; general population). (ii) <i>Exact Duplicate Rows (1,081 duplicates):</i> Retained. Deleting duplicates would destroy 19 legitimate fraud cases (3.86% of all positives). (iii) <i>Feature Representation:</i> 29 features (<i>V</i><sub>1</sub>--<i>V</i><sub>28</sub> plus ln(1 + Amount)). `Time` and `Class` are excluded. (iv) <i>Leak-Free Scaling:</i> `StandardScaler` fitted strictly on 170,528 legitimate training rows.",
        body_style
    ))

    # TABLE I: DATASET SPLIT SUMMARY
    table_split_data = [
        [Paragraph("<b>Partition</b>", table_cell_bold), Paragraph("<b>Total Rows</b>", table_cell_bold), Paragraph("<b>Legit (y=0)</b>", table_cell_bold), Paragraph("<b>Fraud (y=1)</b>", table_cell_bold), Paragraph("<b>Prevalence</b>", table_cell_bold), Paragraph("<b>Role</b>", table_cell_bold)],
        [Paragraph("Train", table_cell), Paragraph("170,888", table_cell), Paragraph("170,528", table_cell), Paragraph("360", table_cell), Paragraph("0.2107%", table_cell), Paragraph("Fitting (y=0 only)", table_cell)],
        [Paragraph("Validation", table_cell), Paragraph("56,957", table_cell), Paragraph("56,900", table_cell), Paragraph("57", table_cell), Paragraph("0.1001%", table_cell), Paragraph("Selection & &tau; freeze", table_cell)],
        [Paragraph("Test (Held-out)", table_cell), Paragraph("56,962", table_cell), Paragraph("56,887", table_cell), Paragraph("75", table_cell), Paragraph("0.1317%", table_cell), Paragraph("One-time benchmark", table_cell)],
        [Paragraph("<b>Total</b>", table_cell_bold), Paragraph("<b>284,807</b>", table_cell_bold), Paragraph("<b>284,315</b>", table_cell_bold), Paragraph("<b>492</b>", table_cell_bold), Paragraph("<b>0.1727%</b>", table_cell_bold), Paragraph("Preserved Full Set", table_cell)]
    ]
    t_split = Table(table_split_data, colWidths=[50, 44, 44, 38, 40, 38])
    t_split.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f0f0")),
        ('LINEABOVE', (0,0), (-1,0), 1.0, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 1.0, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(Spacer(1, 2))
    story.append(Paragraph("TABLE I<br/>CHRONOLOGICAL DATASET PARTITIONING (STRATEGY B)", table_title_style))
    story.append(t_split)
    story.append(Spacer(1, 5))

    # ==================== SECTION V: PHASE 1 GMM ====================
    story.append(Paragraph("V. PHASE 1: GAUSSIAN MIXTURE DENSITY SCORING", sec_heading))
    story.append(Paragraph(
        "<b>Baseline Formulation (K=1, Full Covariance):</b> The single-component baseline fits a 29-dimensional Gaussian on legitimate training rows (464 parameters). With covariance regularizer reg_covar = 10<sup>&minus;6</sup>, the covariance condition number is &kappa; = 6.475, establishing an exceptionally well-conditioned density reference.",
        body_style
    ))
    story.append(Paragraph(
        "<b>12-Model Grid Search:</b> We evaluated <i>K</i> &isin; {2, 3, 4, 5} across `full`, `diag`, and `tied` covariance geometries (reg_covar = 10<sup>&minus;6</sup>, <i>n</i>_init = 5, seed 42) on legitimate training rows.",
        body_style
    ))

    # TABLE II: GMM GRID
    table_gmm_data = [
        [Paragraph("<b>Model</b>", table_cell_bold), Paragraph("<b>K</b>", table_cell_bold), Paragraph("<b>Cov.</b>", table_cell_bold), Paragraph("<b>Params</b>", table_cell_bold), Paragraph("<b>BIC</b>", table_cell_bold), Paragraph("<b>Val AP</b>", table_cell_bold), Paragraph("<b>Recall@1%</b>", table_cell_bold)],
        [Paragraph("gmm_baseline", table_cell), Paragraph("1", table_cell), Paragraph("full", table_cell), Paragraph("464", table_cell), Paragraph("13.75M", table_cell), Paragraph("0.0740", table_cell), Paragraph("73.68%", table_cell)],
        [Paragraph("gmm_k2_full", table_cell), Paragraph("2", table_cell), Paragraph("full", table_cell), Paragraph("929", table_cell), Paragraph("5.97M", table_cell), Paragraph("0.0750", table_cell), Paragraph("75.44%", table_cell)],
        [Paragraph("gmm_k2_diag", table_cell), Paragraph("2", table_cell), Paragraph("diag", table_cell), Paragraph("88", table_cell), Paragraph("11.97M", table_cell), Paragraph("0.0455", table_cell), Paragraph("57.89%", table_cell)],
        [Paragraph("gmm_k2_tied", table_cell), Paragraph("2", table_cell), Paragraph("tied", table_cell), Paragraph("494", table_cell), Paragraph("13.41M", table_cell), Paragraph("0.0742", table_cell), Paragraph("73.68%", table_cell)],
        [Paragraph("gmm_k3_full", table_cell), Paragraph("3", table_cell), Paragraph("full", table_cell), Paragraph("1,394", table_cell), Paragraph("3.92M", table_cell), Paragraph("0.0828", table_cell), Paragraph("75.44%", table_cell)],
        [Paragraph("gmm_k3_diag", table_cell), Paragraph("3", table_cell), Paragraph("diag", table_cell), Paragraph("147", table_cell), Paragraph("11.27M", table_cell), Paragraph("0.0533", table_cell), Paragraph("66.67%", table_cell)],
        [Paragraph("<b>gmm_k4_full</b>", table_cell_bold), Paragraph("<b>4</b>", table_cell_bold), Paragraph("<b>full</b>", table_cell_bold), Paragraph("<b>1,859</b>", table_cell_bold), Paragraph("<b>-4.82M</b>", table_cell_bold), Paragraph("<b>0.4840</b>", table_cell_bold), Paragraph("<b>75.44%</b>", table_cell_bold)],
        [Paragraph("gmm_k4_diag", table_cell), Paragraph("4", table_cell), Paragraph("diag", table_cell), Paragraph("206", table_cell), Paragraph("10.93M", table_cell), Paragraph("0.0591", table_cell), Paragraph("68.42%", table_cell)],
        [Paragraph("gmm_k4_tied", table_cell), Paragraph("4", table_cell), Paragraph("tied", table_cell), Paragraph("554", table_cell), Paragraph("11.82M", table_cell), Paragraph("0.0723", table_cell), Paragraph("75.44%", table_cell)],
        [Paragraph("gmm_k5_full", table_cell), Paragraph("5", table_cell), Paragraph("full", table_cell), Paragraph("2,324", table_cell), Paragraph("-4.76M", table_cell), Paragraph("0.2192", table_cell), Paragraph("73.68%", table_cell)],
        [Paragraph("gmm_k5_tied", table_cell), Paragraph("5", table_cell), Paragraph("tied", table_cell), Paragraph("584", table_cell), Paragraph("12.02M", table_cell), Paragraph("0.1770", table_cell), Paragraph("75.44%", table_cell)],
    ]
    t_gmm = Table(table_gmm_data, colWidths=[52, 18, 28, 32, 38, 38, 46])
    t_gmm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f0f0")),
        ('BACKGROUND', (0,7), (-1,7), colors.HexColor("#e6f2ff")),
        ('LINEABOVE', (0,0), (-1,0), 1.0, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 1.0, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.6),
        ('TOPPADDING', (0,0), (-1,-1), 1.6),
    ]))
    story.append(Spacer(1, 2))
    story.append(Paragraph("TABLE II<br/>PHASE 1: GMM MODEL SELECTION GRID (VALIDATION EVALUATION)", table_title_style))
    story.append(t_gmm)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Key Findings (RQ2, RQ3):</b><br/>"
        "1) <i>Diagonal Covariance Deficit:</i> Diagonal models (`diag`) fail across all metrics (AP &approx; 0.045&ndash;0.059). Assuming features are conditionally independent discards critical pairwise correlation between PCA dimensions and transaction Amount.<br/>"
        "2) <i>Negative BIC in Continuous Space:</i> Continuous density models can have <i>p</i>(<i>x</i>) > 1 in dense clusters. When 2 ln <i>L&#770;</i> exceeds <i>p</i> ln <i>N</i>, BIC becomes negative; more negative values denote tighter fit.<br/>"
        "3) <i>The Champion Candidate:</i> `gmm_k4_full` achieves BIC = &minus;4.82M and validation AP = 0.4840 (6.5&times; baseline gain).",
        body_style
    ))

    # Embed Figure: BIC/AIC Curves
    bic_img_path = "results/figures/bic_aic_curve.png"
    if os.path.exists(bic_img_path):
        story.append(Spacer(1, 3))
        story.append(Image(bic_img_path, width=COL_WIDTH - 6, height=(COL_WIDTH - 6) * 0.44))
        story.append(Paragraph("Fig. 1. AIC and BIC goodness-of-fit curves across component counts <i>K</i> and covariance structures. Full covariance dramatically outperforms diagonal.", fig_caption))

    # ==================== SECTION VI: PHASE 2 AUTOENCODER ====================
    story.append(Paragraph("VI. PHASE 2: DEEP AUTOENCODER ANOMALY DETECTION", sec_heading))
    story.append(Paragraph(
        "<b>Architecture:</b> To benchmark neural reconstruction against probabilistic density, we designed a symmetric 5-layer MLP in PyTorch:<br/>"
        "&nbsp;&nbsp;29 &rarr; Linear(20) &rarr; ReLU &rarr; Linear(10) &rarr; ReLU &rarr; Linear(5) [Bottleneck]<br/>"
        "&nbsp;&nbsp;&rarr; Linear(10) &rarr; ReLU &rarr; Linear(20) &rarr; ReLU &rarr; Linear(29) [Linear Output]<br/>"
        "Comprising 1,929 trainable parameters, the network compresses 29 features by 5.8&times; into a 5D latent space. A <b>Linear output activation</b> is required: standardized Z-score inputs have unbounded real ranges (&minus;&infin; to +&infin;); Sigmoid or ReLU activations would clip negative values and distort reconstruction.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Training Protocol:</b> Trained on CPU with Adam (lr = 0.001, batch size 256, max 30 epochs, seed 42) strictly on 170,528 legitimate training rows. Early stopping was monitored on 56,900 legitimate validation rows (patience 5). Training completed in 10.24 seconds, reaching training MSE = 0.4658 and validation MSE = 0.7014 at epoch 30.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Testing Hypotheses H1 & H2:</b> Validation AP reached 0.0765, exceeding the GMM baseline (0.0740), supporting <b>H1</b>. However, while reconstruction MSE avoided GMM covariance singularity (supporting <b>H2</b>), it did not match the specialized peak AP of the 4-component full-covariance GMM (0.4840).",
        body_style
    ))

    # Embed Figure: Score Distributions
    dist_img_path = "results/figures/validation_score_distributions.png"
    if os.path.exists(dist_img_path):
        story.append(Spacer(1, 3))
        story.append(Image(dist_img_path, width=COL_WIDTH - 6, height=(COL_WIDTH - 6) * 0.44))
        story.append(Paragraph("Fig. 2. Log-scale anomaly score distributions: legitimate transactions (blue) vs. fraudulent transactions (crimson), showing clear separation.", fig_caption))

    # ==================== SECTION VII: HELD-OUT TEST BENCHMARK ====================
    story.append(Paragraph("VII. CROSS-MODEL BENCHMARK & HELD-OUT TEST EVALUATION", sec_heading))
    story.append(Paragraph(
        "<b>Held-Out Test Set Protocol:</b> The 56,962-row test partition (75 frauds) was evaluated exactly once. Neither models nor preprocessing were refitted. Operating thresholds for 1% and 5% FPR targets derived from the validation partition were frozen prior to test evaluation.",
        body_style
    ))

    # TABLE III: TEST BENCHMARK
    table_test_data = [
        [Paragraph("<b>Metric</b>", table_cell_bold), Paragraph("<b>GMM Baseline (K=1)</b>", table_cell_bold), Paragraph("<b>GMM K=4 Full</b>", table_cell_bold), Paragraph("<b>Autoencoder Baseline</b>", table_cell_bold)],
        [Paragraph("Score Function", table_cell), Paragraph("NLL: &minus;ln <i>p</i>(<i>x</i>)", table_cell), Paragraph("NLL: &minus;ln <i>p</i>(<i>x</i>)", table_cell), Paragraph("Reconstruction MSE", table_cell)],
        [Paragraph("Parameters", table_cell), Paragraph("464", table_cell), Paragraph("1,859", table_cell), Paragraph("1,929", table_cell)],
        [Paragraph("<b>Average Precision</b>", table_cell_bold), Paragraph("<b>0.1610</b>", table_cell_bold), Paragraph("<b>0.5770</b>", table_cell_bold), Paragraph("<b>0.1793</b>", table_cell_bold)],
        [Paragraph("95% Bootstrap CI", table_cell), Paragraph("[0.121, 0.221]", table_cell), Paragraph("[0.459, 0.697]", table_cell), Paragraph("[0.132, 0.245]", table_cell)],
        [Paragraph("ROC-AUC", table_cell), Paragraph("0.9622", table_cell), Paragraph("0.9156", table_cell), Paragraph("0.9436", table_cell)],
        [Paragraph("<b>Recall @ 1% &tau;<sub>val</sub></b>", table_cell_bold), Paragraph("<b>72.00% (54/75)</b>", table_cell_bold), Paragraph("<b>76.00% (57/75)</b>", table_cell_bold), Paragraph("<b>73.33% (55/75)</b>", table_cell_bold)],
        [Paragraph("Realized Test FPR", table_cell), Paragraph("0.66% (378 FP)", table_cell), Paragraph("0.59% (335 FP)", table_cell), Paragraph("0.56% (318 FP)", table_cell)],
        [Paragraph("Total Alerts (Precision)", table_cell), Paragraph("432 (12.50%)", table_cell), Paragraph("392 (14.54%)", table_cell), Paragraph("373 (14.75%)", table_cell)],
        [Paragraph("<b>Recall @ 5% &tau;<sub>val</sub></b>", table_cell_bold), Paragraph("<b>84.00% (63/75)</b>", table_cell_bold), Paragraph("<b>80.00% (60/75)</b>", table_cell_bold), Paragraph("<b>80.00% (60/75)</b>", table_cell_bold)],
        [Paragraph("Realized Test FPR", table_cell), Paragraph("4.01% (2,283 FP)", table_cell), Paragraph("3.68% (2,096 FP)", table_cell), Paragraph("3.60% (2,049 FP)", table_cell)],
    ]
    t_test = Table(table_test_data, colWidths=[64, 62, 64, 64])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f0f0")),
        ('LINEABOVE', (0,0), (-1,0), 1.0, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 1.0, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.6),
        ('TOPPADDING', (0,0), (-1,-1), 1.6),
    ]))
    story.append(Spacer(1, 2))
    story.append(Paragraph("TABLE III<br/>FINAL HELD-OUT TEST BENCHMARK (N=56,962, 75 FRAUDS)", table_title_style))
    story.append(t_test)
    story.append(Spacer(1, 4))

    # Embed Figure: PR Curve
    pr_img_path = "results/figures/final_test_precision_recall_curve.png"
    if os.path.exists(pr_img_path):
        story.append(Spacer(1, 3))
        story.append(Image(pr_img_path, width=COL_WIDTH - 6, height=(COL_WIDTH - 6) * 0.44))
        story.append(Paragraph("Fig. 3. Final Precision-Recall curves on the held-out test partition. GMM K=4 Full maintains dominant precision across operative recall ranges.", fig_caption))

    # TABLE IV: TOP-N CAPACITY
    table_topn_data = [
        [Paragraph("<b>Budget</b>", table_cell_bold), Paragraph("<b>Metric</b>", table_cell_bold), Paragraph("<b>GMM Base</b>", table_cell_bold), Paragraph("<b>GMM K=4</b>", table_cell_bold), Paragraph("<b>Autoencoder</b>", table_cell_bold)],
        [Paragraph("<b>Top 50</b>", table_cell), Paragraph("Frauds Caught<br/>Precision", table_cell), Paragraph("9 / 75<br/>18.00%", table_cell), Paragraph("<b>39 / 75<br/>78.00%</b>", table_cell_bold), Paragraph("11 / 75<br/>22.00%", table_cell)],
        [Paragraph("<b>Top 100</b>", table_cell), Paragraph("Frauds Caught<br/>Precision", table_cell), Paragraph("28 / 75<br/>28.00%", table_cell), Paragraph("<b>49 / 75<br/>49.00%</b>", table_cell_bold), Paragraph("31 / 75<br/>31.00%", table_cell)],
        [Paragraph("<b>Top 200</b>", table_cell), Paragraph("Frauds Caught<br/>Precision", table_cell), Paragraph("38 / 75<br/>19.00%", table_cell), Paragraph("<b>55 / 75<br/>27.50%</b>", table_cell_bold), Paragraph("40 / 75<br/>20.00%", table_cell)],
        [Paragraph("<b>Top 500</b>", table_cell), Paragraph("Frauds Caught<br/>Precision", table_cell), Paragraph("56 / 75<br/>11.20%", table_cell), Paragraph("<b>57 / 75<br/>11.40%</b>", table_cell_bold), Paragraph("56 / 75<br/>11.20%", table_cell)],
    ]
    t_topn = Table(table_topn_data, colWidths=[38, 64, 50, 51, 51])
    t_topn.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f0f0f0")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#f8fbff")),
        ('LINEABOVE', (0,0), (-1,0), 1.0, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 1.0, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.6),
        ('TOPPADDING', (0,0), (-1,-1), 1.6),
    ]))
    story.append(Spacer(1, 2))
    story.append(Paragraph("TABLE IV<br/>FRAUD INTERCEPTION UNDER FIXED OPERATIONAL CAPACITY", table_title_style))
    story.append(t_topn)
    story.append(Spacer(1, 4))

    # Embed Figure: Review Capacity Curves
    cap_img_path = "results/figures/final_test_review_capacity.png"
    if os.path.exists(cap_img_path):
        story.append(Spacer(1, 3))
        story.append(Image(cap_img_path, width=COL_WIDTH - 6, height=(COL_WIDTH - 6) * 0.44))
        story.append(Paragraph("Fig. 4. Cumulative fraud interception as a function of daily analyst review capacity on the held-out test partition.", fig_caption))

    # ==================== SECTION VIII: DISCUSSION ====================
    story.append(Paragraph("VIII. DISCUSSION AND OPERATIONAL DEPLOYMENT", sec_heading))
    story.append(Paragraph(
        "<b>The Two-Tier Triage Pipeline (RQ9):</b> Real-world fraud desks face opposing constraints: real-time automated step-up challenges require high precision to avoid cardholder friction, while offline investigation queues demand high recall. Empirical findings show no single model optimizes both:",
        body_style
    ))
    story.append(Paragraph(
        "&bull; <i>Tier 1 (High-Precision Rapid Interception):</i> Employs `GMM K=4 Full` on a dynamic capacity quota (Top 50&ndash;100 alerts). It intercepts 39 of 75 frauds within just the first 50 alerts (<b>78.00% precision</b>), triggering automated SMS OTP verification with minimal friction.<br/>"
        "&bull; <i>Tier 2 (Broad High-Recall Safety Net):</i> Employs `GMM Baseline (K=1)` at the frozen 5% FPR threshold (&tau; = 60.726). It intercepts <b>84.00% of all fraud (63/75)</b>, routing severe multivariate tail anomalies to an offline review queue.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Density vs. Subspace Reconstruction:</b> GMMs scale distances by the inverted covariance matrix (&Sigma;<sup>&minus;1</sup>), penalizing low-probability tail points. The Autoencoder projects onto a continuous 5D non-linear manifold; it achieves robust baseline performance (0.1793 AP) without EM covariance condition sensitivity, but lacks the sharp local density peaks that multi-component GMMs exploit.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Threats to Validity:</b><br/>"
        "&bull; <i>Single Chronological Horizon:</i> The dataset spans 48 hours. Long-term seasonal shifts (holiday spending, month-end billing) were not observed.<br/>"
        "&bull; <i>Sample Variance on Positives:</i> With <i>N</i> = 75 test frauds, catching 3 more frauds changes recall by 4.0 pp; bootstrap confidence intervals reflect this uncertainty.<br/>"
        "&bull; <i>EM Seed Sensitivity:</i> While the seed-42 <i>K</i> = 4 GMM generalized to 0.5770 AP, alternative EM seeds produced lower validation AP (0.07&ndash;0.10). The unimodal baseline provides complete seed invariance.",
        body_style
    ))

    # ==================== SECTION IX: CONCLUSION ====================
    story.append(Paragraph("IX. CONCLUSION AND FUTURE WORK", sec_heading))
    story.append(Paragraph(
        "Under a leak-free chronological protocol, unsupervised anomaly detection delivers effective fraud triage without exposing models to fraud labels during training. A 4-component full-covariance GMM provides dominant precision under constrained capacity (0.5770 test AP, 78% Top-50 precision), while a single-component baseline delivers superior broad-spectrum recall (84%). Deep autoencoders provide competitive baseline performance (0.1793 test AP) without EM numerical instability. Future work will explore semi-supervised latent adaptation using confirmed analyst feedback and streaming covariance updates.",
        body_style
    ))

    # ==================== REFERENCES ====================
    story.append(Spacer(1, 3))
    story.append(Paragraph("REFERENCES", sec_heading))
    refs = [
        "[1] A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, \"Calibrating probability with undersampling for unbalanced classification,\" in <i>Proc. IEEE SSCI</i>, 2015, pp. 159&ndash;166.",
        "[2] M. Ahmed, A. N. Mahmood, and J. Hu, \"A survey of network anomaly detection techniques,\" <i>J. Netw. Comput. Appl.</i>, vol. 60, pp. 19&ndash;31, 2016.",
        "[3] J. Davis and M. Goadrich, \"The relationship between Precision-Recall and ROC curves,\" in <i>Proc. ICML</i>, 2006, pp. 233&ndash;240.",
        "[4] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, \"SMOTE: Synthetic minority over-sampling technique,\" <i>J. Artif. Intell. Res.</i>, vol. 16, pp. 321&ndash;357, 2002.",
        "[5] V. Chandola, A. Banerjee, and V. Kumar, \"Anomaly detection: A survey,\" <i>ACM Comput. Surv.</i>, vol. 41, no. 3, pp. 1&ndash;58, 2009.",
        "[6] D. A. Reynolds, \"Gaussian mixture models,\" <i>Encycl. Biom.</i>, vol. 741, pp. 659&ndash;663, 2009.",
        "[7] J. An and S. Cho, \"Variational autoencoder based anomaly detection using reconstruction probability,\" <i>SNU Data Mining Center Tech. Rep.</i>, 2015."
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Publication-quality PDF generated successfully: {output_path}")

if __name__ == "__main__":
    out_pdf = "reports/card_fraud_triage_paper.pdf"
    create_paper_pdf(out_pdf)
    import shutil
    shutil.copy(out_pdf, "paper.pdf")
    print("Copied to paper.pdf in root workspace.")
