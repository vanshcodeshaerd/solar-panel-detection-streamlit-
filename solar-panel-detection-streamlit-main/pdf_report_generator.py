import os
import streamlit as st
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import tempfile
import json
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import zipfile
import base64
from io import BytesIO
import re

class PDFReportGenerator:
    """Comprehensive PDF report generator for solar panel detection"""
    
    def __init__(self, reports_dir="./reports"):
        self.reports_dir = reports_dir
        self.ensure_reports_directory()
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def ensure_reports_directory(self):
        """Create reports directory if it doesn't exist"""
        os.makedirs(self.reports_dir, exist_ok=True)
        # Create subdirectories for organization
        os.makedirs(os.path.join(self.reports_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, "archive"), exist_ok=True)
    
    def setup_custom_styles(self):
        """Setup custom styles for PDF report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            borderWidth=0,
            borderColor=colors.white
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkgreen,
            alignment=TA_LEFT,
            borderWidth=0
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue,
            alignment=TA_LEFT,
            borderWidth=0
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leading=14,
            alignment=TA_LEFT
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='CustomFooter',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        ))
    
    def generate_filename(self, prefix="report", file_extension=".pdf"):
        """Generate unique filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = datetime.now().strftime("%f")[-6:]  # microseconds for uniqueness
        return f"{prefix}_{timestamp}_{unique_id}{file_extension}"
    
    def save_analysis_image(self, image, filename):
        """Save analysis image to reports directory"""
        image_path = os.path.join(self.reports_dir, "images", filename)
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save with high quality
        image.save(image_path, "JPEG", quality=95, optimize=True)
        return image_path
    
    def create_detection_summary_table(self, detection_results):
        """Create summary table for detection results"""
        data = [
            ['Metric', 'Value', 'Status'],
            ['Total Detections', str(detection_results.get('total_detections', 0)), ''],
            ['Solar Panels', str(detection_results.get('solar_panels', 0)), '✅' if detection_results.get('solar_panels', 0) > 0 else '❌'],
            ['Non-Solar Rooftops', str(detection_results.get('non_solar_rooftops', 0)), ''],
            ['Average Confidence', f"{detection_results.get('avg_confidence', 0):.1f}%", ''],
            ['Detection Coverage', f"{detection_results.get('coverage_percentage', 0):.1f}%", ''],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            
            # Data rows
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        return table
    
    def create_detailed_detections_table(self, predictions):
        """Create detailed table for individual detections"""
        if not predictions:
            return Paragraph("No detections found.", self.styles['CustomBody'])
        
        data = [['#', 'Type', 'Confidence', 'X', 'Y', 'Width', 'Height']]
        
        for i, pred in enumerate(predictions, 1):
            confidence = pred.get('confidence', 0) * 100
            x = pred.get('x', 0)
            y = pred.get('y', 0)
            width = pred.get('width', 0)
            height = pred.get('height', 0)
            
            # Determine detection type
            detection_type = "Solar Panel" if confidence > 50 else "Rooftop"
            
            data.append([
                str(i),
                detection_type,
                f"{confidence:.1f}%",
                f"{x:.0f}",
                f"{y:.0f}",
                f"{width:.0f}",
                f"{height:.0f}"
            ])
        
        table = Table(data, colWidths=[0.5*inch, 1.2*inch, 1*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            
            # Data rows
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        return table
    
    def generate_pdf_report(self, original_image, analysis_image, detection_results, predictions, metadata=None):
        """Generate comprehensive PDF report"""
        
        # Generate unique filename
        pdf_filename = self.generate_filename("solar_detection_report")
        pdf_path = os.path.join(self.reports_dir, pdf_filename)
        
        # Create the PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Build story (content)
        story = []
        
        # Title Page
        story.append(Paragraph("Solar Panel Detection Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Generated on:</b> {report_date}", self.styles['CustomBody']))
        story.append(Paragraph(f"<b>Report ID:</b> {pdf_filename.replace('.pdf', '')}", self.styles['CustomBody']))
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomSubtitle']))
        
        # Create summary based on results
        solar_count = detection_results.get('solar_panels', 0)
        total_count = detection_results.get('total_detections', 0)
        
        if solar_count > 0:
            summary_text = f"""
            The analysis successfully identified <b>{solar_count}</b> solar panel installations 
            out of <b>{total_count}</b> total rooftop detections in the provided image. 
            The detection confidence averaged <b>{detection_results.get('avg_confidence', 0):.1f}%</b>, 
            indicating reliable identification of solar infrastructure.
            """
        else:
            summary_text = f"""
            The analysis processed <b>{total_count}</b> rooftop areas but did not identify 
            any solar panel installations. This location may represent a potential opportunity 
            for solar energy development. The detection confidence averaged 
            <b>{detection_results.get('avg_confidence', 0):.1f}%</b>.
            """
        
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        # Detection Summary Table
        story.append(Paragraph("Detection Summary", self.styles['CustomSubtitle']))
        summary_table = self.create_detection_summary_table(detection_results)
        story.append(summary_table)
        story.append(Spacer(1, 30))
        
        # Visual Analysis
        story.append(PageBreak())
        story.append(Paragraph("Visual Analysis", self.styles['CustomSubtitle']))
        
        # Save and include analysis image
        image_filename = f"analysis_{pdf_filename.replace('.pdf', '.jpg')}"
        image_path = self.save_analysis_image(analysis_image, image_filename)
        
        try:
            # Add analysis image to PDF
            img = RLImage(image_path, width=6*inch, height=4.5*inch)
            story.append(img)
            story.append(Spacer(1, 20))
            
            # Image caption
            story.append(Paragraph("Figure 1: Solar Panel Detection Analysis with Classification Markers", 
                                self.styles['CustomBody']))
        except Exception as e:
            story.append(Paragraph(f"Note: Analysis image could not be embedded: {str(e)}", 
                                self.styles['CustomBody']))
        
        story.append(Spacer(1, 30))
        
        # Detailed Detection Results
        story.append(Paragraph("Detailed Detection Results", self.styles['CustomSubtitle']))
        
        if predictions:
            detailed_table = self.create_detailed_detections_table(predictions)
            story.append(detailed_table)
        else:
            story.append(Paragraph("No detailed detection data available.", self.styles['CustomBody']))
        
        story.append(Spacer(1, 30))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.styles['CustomSubtitle']))
        
        if solar_count > 0:
            recommendations = """
            <b>For Existing Solar Installations:</b><br/>
            • Monitor panel performance regularly<br/>
            • Schedule periodic maintenance and cleaning<br/>
            • Consider energy storage integration<br/>
            • Monitor for shading issues from vegetation growth<br/><br/>
            
            <b>For Non-Solar Rooftops:</b><br/>
            • Evaluate remaining rooftop areas for expansion<br/>
            • Conduct feasibility studies for additional installations<br/>
            • Consider energy consumption patterns for sizing
            """
        else:
            recommendations = """
            <b>Solar Development Opportunity:</b><br/>
            • Conduct professional solar feasibility assessment<br/>
            • Evaluate roof structure and orientation<br/>
            • Analyze energy consumption patterns<br/>
            • Investigate available incentives and rebates<br/>
            • Consider environmental impact and benefits<br/><br/>
            
            <b>Next Steps:</b><br/>
            • Contact solar installation providers<br/>
            • Request detailed site assessment<br/>
            • Review financing options and ROI projections
            """
        
        story.append(Paragraph(recommendations, self.styles['CustomBody']))
        story.append(Spacer(1, 30))
        
        # Technical Information
        if metadata:
            story.append(Paragraph("Technical Information", self.styles['CustomSubtitle']))
            
            tech_info = f"""
            <b>Image Processing:</b><br/>
            • Original Image Size: {metadata.get('original_size', 'N/A')}<br/>
            • Processing Time: {metadata.get('processing_time', 'N/A')}<br/>
            • Model Version: {metadata.get('model_version', 'N/A')}<br/>
            • Confidence Threshold: {metadata.get('confidence_threshold', 'N/A')}<br/><br/>
            
            <b>Detection Parameters:</b><br/>
            • Analysis Type: Automated AI Detection<br/>
            • Classification: Solar Panel vs Non-Solar Rooftop<br/>
            • Coverage Area: {detection_results.get('coverage_percentage', 0):.1f}%<br/>
            • Total Detections: {detection_results.get('total_detections', 0)}
            """
            
            story.append(Paragraph(tech_info, self.styles['CustomBody']))
        
        # Footer
        story.append(PageBreak())
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("Generated by Solar Panel Detection System", self.styles['CustomFooter']))
        story.append(Paragraph(f"Report ID: {pdf_filename.replace('.pdf', '')}", self.styles['CustomFooter']))
        story.append(Paragraph(f"Page <seq id='page'/>", self.styles['CustomFooter']))
        
        # Build PDF
        try:
            doc.build(story)
            return {
                'success': True,
                'pdf_path': pdf_path,
                'pdf_filename': pdf_filename,
                'message': f"PDF report generated successfully: {pdf_filename}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to generate PDF: {str(e)}"
            }
    
    def get_download_link(self, pdf_path, filename=None):
        """Generate download link for PDF file"""
        if not os.path.exists(pdf_path):
            return None
        
        if filename is None:
            filename = os.path.basename(pdf_path)
        
        # Read file and encode
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        # Encode to base64
        b64 = base64.b64encode(pdf_bytes).decode()
        
        # Create download link
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📥 Download PDF Report</a>'
        return href
    
    def list_reports(self, limit=10):
        """List available PDF reports"""
        reports = []
        
        if os.path.exists(self.reports_dir):
            for file in os.listdir(self.reports_dir):
                if file.endswith('.pdf'):
                    file_path = os.path.join(self.reports_dir, file)
                    stat = os.stat(file_path)
                    reports.append({
                        'filename': file,
                        'path': file_path,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime),
                        'download_link': self.get_download_link(file_path, file)
                    })
        
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x['created'], reverse=True)
        return reports[:limit]
    
    def cleanup_old_reports(self, days_old=7):
        """Clean up old reports to save space"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        cleaned = 0
        
        if os.path.exists(self.reports_dir):
            for file in os.listdir(self.reports_dir):
                if file.endswith('.pdf'):
                    file_path = os.path.join(self.reports_dir, file)
                    if os.path.getmtime(file_path) < cutoff_time:
                        try:
                            os.remove(file_path)
                            cleaned += 1
                        except Exception:
                            pass
        
        return cleaned

# Streamlit integration helper
def create_pdf_download_section(pdf_result):
    """Create Streamlit section for PDF download"""
    if pdf_result['success']:
        st.success("✅ " + pdf_result['message'])
        
        # Provide download link
        if os.path.exists(pdf_result['pdf_path']):
            with open(pdf_result['pdf_path'], "rb") as f:
                pdf_bytes = f.read()
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=pdf_result['pdf_filename'],
                mime="application/pdf",
                help="Click to download the complete analysis report"
            )
            
            # Show file info
            file_size = os.path.getsize(pdf_result['pdf_path']) / 1024  # KB
            st.info(f"📄 File size: {file_size:.1f} KB")
    else:
        st.error("❌ " + pdf_result['message'])

# Example usage
if __name__ == "__main__":
    # Test the PDF generator
    generator = PDFReportGenerator()
    
    # Create dummy data for testing
    from PIL import Image, ImageDraw
    import numpy as np
    
    # Create test image
    test_image = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(test_image)
    draw.rectangle([100, 100, 300, 200], fill='green', outline='black')
    draw.rectangle([400, 150, 600, 250], fill='red', outline='black')
    
    # Test data
    detection_results = {
        'total_detections': 2,
        'solar_panels': 1,
        'non_solar_rooftops': 1,
        'avg_confidence': 85.5,
        'coverage_percentage': 75.2
    }
    
    predictions = [
        {'confidence': 0.92, 'x': 200, 'y': 150, 'width': 200, 'height': 100},
        {'confidence': 0.78, 'x': 500, 'y': 200, 'width': 200, 'height': 100}
    ]
    
    metadata = {
        'original_size': '800x600',
        'processing_time': '2.3 seconds',
        'model_version': 'v2.1',
        'confidence_threshold': '50%'
    }
    
    # Generate PDF
    result = generator.generate_pdf_report(
        original_image=test_image,
        analysis_image=test_image,
        detection_results=detection_results,
        predictions=predictions,
        metadata=metadata
    )
    
    print("PDF Generation Result:", result)
