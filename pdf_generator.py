import io
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(prop_details, prediction_res, locality_stats, top_factors):
    """
    Generates a formal PriceWise PDF report for a property valuation in Bengaluru using ReportLab.
    Returns bytes of the PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=2
    )
    
    tagline_style = ParagraphStyle(
        'DocTagline',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#2563EB'),
        fontName='Helvetica-Bold',
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155')
    )

    story = []
    
    # Header Branding
    story.append(Paragraph("PriceWise", title_style))
    story.append(Paragraph("Predict smarter. Invest better.", tagline_style))
    story.append(Paragraph("Bengaluru Real Estate AI Property Evaluation Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=15))
    
    # Prediction Summary Card Table
    pred_data = [
        [Paragraph("<b>ESTIMATED FAIR MARKET VALUE</b>", ParagraphStyle('WH', parent=body_style, textColor=colors.HexColor('#475569'))),
         Paragraph("<b>PRICE RANGE (95% CONFIDENCE)</b>", ParagraphStyle('WH', parent=body_style, textColor=colors.HexColor('#475569')))],
        [Paragraph(f"<font size=18 color='#2563EB'><b>₹ {prediction_res['price_lakhs']:.2f} Lakhs</b></font><br/><font size=9 color='#64748B'>({prediction_res['price_crores']:.2f} Cr)</font>", body_style),
         Paragraph(f"<font size=14 color='#16A34A'><b>₹ {prediction_res['lower']:.2f} L – ₹ {prediction_res['upper']:.2f} L</b></font>", body_style)],
        [Paragraph(f"<b>Estimated Rate:</b> ₹ {prediction_res['price_per_sqft']:,.0f} / sq.ft.", body_style),
         Paragraph(f"<b>Model Engine:</b> {prediction_res['model_name']}", body_style)]
    ]
    
    pred_table = Table(pred_data, colWidths=[270, 270])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 15))
    
    # Property Specifications
    story.append(Paragraph("Bengaluru Property Specifications", h2_style))
    specs_data = [
        ["Attribute", "Details", "Attribute", "Details"],
        ["Bengaluru Locality", prop_details.get('location', '-'), "Area Type", prop_details.get('area_type', '-')],
        ["Total Area", f"{prop_details.get('total_sqft', 0):,} sq.ft.", "Locality Tier", locality_stats.get('tier', 'N/A')],
        ["BHK Bedrooms", f"{prop_details.get('bhk', 0)} BHK", "Availability Status", prop_details.get('availability', '-')],
        ["Bathrooms", str(prop_details.get('bath', 0)), "Gated Society", "Yes" if prop_details.get('has_society') else "No"],
        ["Balconies", str(prop_details.get('balcony', 0)), "Locality Benchmark Rate", f"₹ {locality_stats.get('avg_pps', 0):,.0f} / sq.ft."]
    ]
    
    specs_table = Table(specs_data, colWidths=[130, 140, 130, 140])
    specs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EFF6FF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BFDBFE')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTSIZE', (0, 0), (-1, -1), 9)
    ]))
    story.append(specs_table)
    story.append(Spacer(1, 15))
    
    # Key Valuation Drivers
    story.append(Paragraph("Key PriceWise Drivers (Explainability)", h2_style))
    factors_data = [["Rank", "Valuation Driver Factor", "Impact Contribution"]]
    for idx, (factor, val) in enumerate(top_factors[:5], 1):
        factors_data.append([str(idx), factor, f"{val:.1f}% impact"])
        
    factors_table = Table(factors_data, colWidths=[50, 330, 160])
    factors_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F0FDF4')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BBF7D0')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTSIZE', (0, 0), (-1, -1), 9)
    ]))
    story.append(factors_table)
    story.append(Spacer(1, 15))
    
    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=10))
    story.append(Paragraph(
        "<b>Disclaimer:</b> PriceWise report outputs are statistical fair market estimates based on historical machine learning training data for Bengaluru real estate. Predict smarter. Invest better.",
        ParagraphStyle('Disc', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.HexColor('#94A3B8'))
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
