import { jsPDF } from 'jspdf'
import { listChatMessages } from './api'

function stripEmojis(text: string): string {
  return text.replace(/[\u{1F600}-\u{1F64F}]/gu, '') // Emoticons
    .replace(/[\u{1F300}-\u{1F5FF}]/gu, '') // Misc Symbols and Pictographs
    .replace(/[\u{1F680}-\u{1F6FF}]/gu, '') // Transport and Map
    .replace(/[\u{1F1E0}-\u{1F1FF}]/gu, '') // Flags
    .replace(/[\u{2600}-\u{26FF}]/gu, '')   // Misc symbols
    .replace(/[\u{2700}-\u{27BF}]/gu, '')   // Dingbats
    .replace(/[\u{FE00}-\u{FE0F}]/gu, '')   // Variation Selectors
    .replace(/[\u{1F900}-\u{1F9FF}]/gu, '') // Supplemental Symbols and Pictographs
    .replace(/[\u{1FA00}-\u{1FA6F}]/gu, '') // Chess Symbols
    .trim()
}


export async function exportChatAsPDF(chatId: string, chatTitle: string = 'Chat Export') {
  try {
    const msgs = await listChatMessages(chatId)

    const doc = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    })

    const font = 'helvetica'
    const codeFont = 'courier'
    const pageWidth = doc.internal.pageSize.getWidth()
    const pageHeight = doc.internal.pageSize.getHeight()
    const margin = 15
    const contentWidth = pageWidth - (margin * 2)
    let yPosition = margin

    // Helper to add new page
    const checkAndAddPage = (neededSpace: number) => {
      if (yPosition + neededSpace > pageHeight - margin) {
        doc.addPage()
        yPosition = margin
        return true
      }
      return false
    }

    // Title - smaller font
    doc.setFontSize(16)
    doc.setFont(font, 'normal')
    doc.text(chatTitle, margin, yPosition)
    yPosition += 8

    // Date - smaller font
    doc.setFontSize(8)
    doc.setFont(font, 'normal')
    doc.setTextColor(100, 100, 100)
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin, yPosition)
    yPosition += 8

    // Separator line
    doc.setDrawColor(41, 128, 185)
    doc.setLineWidth(0.5)
    doc.line(margin, yPosition, pageWidth - margin, yPosition)
    yPosition += 8

    // Process each message
    for (let i = 0; i < msgs.length; i++) {
      const msg = msgs[i]
      const content = stripEmojis(msg.content || '')
      const isUser = msg.role === 'user'
      checkAndAddPage(20)

      if (isUser) {
        // USER MESSAGE - Chat bubble style, right-aligned
        const timestamp = new Date(msg.createdAt * 1000).toLocaleString()

        // Prepare text
        doc.setFontSize(8)
        doc.setFont(font, 'normal')
        const bubbleMaxWidth = contentWidth * 0.5 // 65% of page width
        const textLines = doc.splitTextToSize(content, bubbleMaxWidth - 8)
        const bubbleHeight = (textLines.length * 4) + 4
        const bubbleWidth = Math.min(bubbleMaxWidth, contentWidth * 0.65)
        const bubbleX = pageWidth - margin - bubbleWidth

        checkAndAddPage(bubbleHeight + 10)

        // Timestamp above bubble (right-aligned)
        doc.setFontSize(7)
        doc.setTextColor(120, 120, 120)
        doc.text(timestamp, pageWidth - margin, yPosition, { align: "right" })
        yPosition += 4

        // Draw rounded rectangle bubble
        doc.setFillColor(220, 252, 231) // Light green background
        doc.setDrawColor(167, 243, 208) // Border color
        doc.setLineWidth(0.3)
        doc.roundedRect(bubbleX, yPosition, bubbleWidth, bubbleHeight, 3, 3, 'FD')

        // Text inside bubble
        doc.setTextColor(0, 0, 0)
        doc.setFontSize(8)
        let textY = yPosition + 5
        textLines.forEach((line: string) => {
          doc.text(line, bubbleX + 4, textY)
          textY += 4
        })
        yPosition += bubbleHeight + 6
      } else {
        // ASSISTANT MESSAGE - Regular left-aligned format
        doc.setFontSize(7)
        doc.setFont(font, 'normal')
        doc.setTextColor(120, 120, 120)
        const timestamp = new Date(msg.createdAt * 1000).toLocaleString()
        doc.text(timestamp, pageWidth - margin, yPosition, { align: 'right' })
        yPosition += 5

        // Parse markdown for assistant messages
        const sections = parseMarkdownForPDF(content)

        doc.setTextColor(0, 0, 0)
        for (const section of sections) {
          checkAndAddPage(section.height || 10)

          switch (section.type) {
            case 'heading1':
              yPosition += 4  // Add margin before heading
              doc.setFontSize(12)
              doc.setTextColor(0, 0, 0)

              if (section.text.includes('**')) {
                yPosition = renderTextWithInlineBold(
                  doc,
                  section.text,
                  margin + 2,
                  yPosition,
                  contentWidth - 4,
                  12
                )
              } else {
                doc.setFont('helvetica', 'bold')
                doc.text(cleanMarkdownFormatting(section.text), margin + 2, yPosition)
              }
              yPosition += 5  // Add margin after heading
              break

            case 'heading2':
              yPosition += 4  // Add margin before heading
              doc.setFontSize(11)
              doc.setTextColor(0, 0, 0)

              if (section.text.includes('**')) {
                yPosition = renderTextWithInlineBold(
                  doc,
                  section.text,
                  margin + 2,
                  yPosition,
                  contentWidth - 4,
                  11
                )
              } else {
                doc.setFont('helvetica', 'bold')
                doc.text(cleanMarkdownFormatting(section.text), margin + 2, yPosition)
              }
              yPosition += 5  // Add margin after heading
              break

            case 'heading3':
              yPosition += 4  // Add margin before heading
              doc.setFontSize(10)
              doc.setTextColor(0, 0, 0)

              if (section.text.includes('**')) {
                yPosition = renderTextWithInlineBold(
                  doc,
                  section.text,
                  margin + 2,
                  yPosition,
                  contentWidth - 4,
                  10
                )
              } else {
                doc.setFont('helvetica', 'bold')
                doc.text(cleanMarkdownFormatting(section.text), margin + 2, yPosition)
              }
              yPosition += 5  // Add margin after heading
              break

            case 'code':
              // Code block with background
              const codeLines = doc.splitTextToSize(section.text, contentWidth - 8)
              const codeHeight = codeLines.length * 4 + 4
              checkAndAddPage(codeHeight)

              doc.setFillColor(240, 240, 240)
              doc.rect(margin + 2, yPosition - 2, contentWidth - 4, codeHeight, 'F')

              doc.setFontSize(7)
              doc.setFont(codeFont, 'normal')
              doc.setTextColor(60, 60, 60)

              let codeY = yPosition + 3
              codeLines.forEach((line: string) => {
                doc.text(line, margin + 4, codeY)
                codeY += 4
              })

              yPosition = codeY + 4
              doc.setFont(font, 'normal')
              doc.setTextColor(0, 0, 0)
              break

            case 'ordered-list':
              doc.setFontSize(8)
              doc.setTextColor(0, 0, 0)

              // Check if contains bold
              if (section.text.includes('**')) {
                checkAndAddPage(6)
                doc.text(`${section.number}.`, margin + 4, yPosition)
                yPosition = renderTextWithInlineBold(
                  doc,
                  section.text,
                  margin + 12,
                  yPosition,
                  contentWidth - 12,
                  8
                )
                yPosition += 1
              } else {
                doc.setFont('helvetica', 'normal')
                const cleaned = cleanMarkdownFormatting(section.text)
                const numberedLines = doc.splitTextToSize(cleaned, contentWidth - 12)
                numberedLines.forEach((line: string, idx: number) => {
                  checkAndAddPage(4)
                  const prefix = idx === 0 ? `${section.number}.` : ''
                  doc.text(prefix, margin + 4, yPosition)
                  doc.text(line, margin + 12, yPosition)
                  yPosition += 4
                })
                yPosition += 1
              }
              break

            case 'unordered-list':
              doc.setFontSize(8)
              doc.setTextColor(0, 0, 0)

              // Check if contains bold
              if (section.text.includes('**')) {
                checkAndAddPage(6)
                doc.text('•', margin + 4, yPosition)
                yPosition = renderTextWithInlineBold(
                  doc,
                  section.text,
                  margin + 10,
                  yPosition,
                  contentWidth - 10,
                  8
                )
                yPosition += 1
              } else {
                doc.setFont('helvetica', 'normal')
                const cleaned = cleanMarkdownFormatting(section.text)
                const bulletLines = doc.splitTextToSize(cleaned, contentWidth - 10)
                bulletLines.forEach((line: string) => {
                  checkAndAddPage(4)
                  doc.text('•', margin + 4, yPosition)
                  doc.text(line, margin + 10, yPosition)
                  yPosition += 4
                })
                yPosition += 1
              }
              break

            case 'bold':
              // This case is no longer needed as we handle inline bold in paragraphs
              doc.setFontSize(8)
              doc.setFont('helvetica', 'bold')
              const boldLines = doc.splitTextToSize(section.text, contentWidth - 4)
              boldLines.forEach((line: string) => {
                checkAndAddPage(4)
                doc.text(line, margin + 2, yPosition)
                yPosition += 4
              })
              break

            default: // paragraph
              doc.setFontSize(8)
              doc.setTextColor(0, 0, 0)

              // Check if line contains bold markers
              if (section.text.includes('**')) {
                checkAndAddPage(8)
                yPosition = renderTextWithInlineBold(
                  doc,
                  section.text,
                  margin + 2,
                  yPosition,
                  contentWidth - 4,
                  8
                )
                yPosition += 1
              } else {
                // Regular text without bold - use standard jsPDF wrapping
                doc.setFont('helvetica', 'normal')
                const cleaned = cleanMarkdownFormatting(section.text)
                const lines = doc.splitTextToSize(cleaned, contentWidth - 4)
                lines.forEach((line: string) => {
                  checkAndAddPage(4)
                  doc.text(line, margin + 2, yPosition)
                  yPosition += 4
                })
                yPosition += 1
              }
              break
          }
        }

        // Message separator (only for assistant)
        if (!isUser) {
          yPosition += 4
          doc.setDrawColor(220, 220, 220)
          doc.setLineWidth(0.3)
          doc.line(margin, yPosition, pageWidth - margin, yPosition)
          yPosition += 6
        }
      }
    }
    // Footer on last page
    doc.setFontSize(7)
    doc.setTextColor(150, 150, 150)
    doc.text('Generated by NeuroChat', pageWidth / 2, pageHeight - 10, { align: 'center' })

    // Save the PDF
    const filename = `${sanitizeFilename(chatTitle)}_${Date.now()}.pdf`
    doc.save(filename)
  } catch (error) {
    console.error('PDF generation error:', error)
    throw new Error('Failed to generate PDF. Please try again.')
  }
}


// Parse markdown into structured sections for better PDF rendering
function parseMarkdownForPDF(markdown: string): Array<{ type: string, text: string, height?: number, number?: string }> {
  const sections: Array<{ type: string, text: string, height?: number, number?: string }> = []
  const lines = markdown.split('\n')

  let inCodeBlock = false
  let codeContent: string[] = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    // Skip horizontal rules (---)
    if (line.trim().match(/^[-*_]{3,}$/)) {
      continue
    }

    // Code blocks
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        sections.push({ type: 'code', text: codeContent.join('\n') })
        codeContent = []
        inCodeBlock = false
      } else {
        inCodeBlock = true
      }
      continue
    }

    if (inCodeBlock) {
      codeContent.push(line)
      continue
    }

    // Headings - remove markdown formatting
    if (line.startsWith('### ')) {
      const text = cleanMarkdownFormatting(line.substring(4).trim())
      if (text) sections.push({ type: 'heading3', text })
    } else if (line.startsWith('## ')) {
      const text = cleanMarkdownFormatting(line.substring(3).trim())
      if (text) sections.push({ type: 'heading2', text })
    } else if (line.startsWith('# ')) {
      const text = cleanMarkdownFormatting(line.substring(2).trim())
      if (text) sections.push({ type: 'heading1', text })
    }
    // Lists - distinguish between ordered and unordered
    else if (line.trim().match(/^\d+\. /)) {
      // Ordered list (numbered)
      const match = line.trim().match(/^(\d+)\.\s+(.*)/)
      if (match) {
        const number = match[1]
        const text = match[2]
        const cleaned = cleanMarkdownFormatting(text)
        if (cleaned) sections.push({ type: 'ordered-list', text: cleaned, number })
      }
    } else if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
      // Unordered list (bulleted)
      const text = line.trim().replace(/^[-*]\s+/, '')
      const cleaned = cleanMarkdownFormatting(text)
      if (cleaned) sections.push({ type: 'unordered-list', text: cleaned })
    }
    // Regular paragraphs
    else if (line.trim()) {
      if (line.trim()) {
        sections.push({ type: 'paragraph', text: line.trim() })
      }
    }
  }

  return sections
}

// Clean markdown formatting from text
function cleanMarkdownFormatting(text: string): string {
  return text
    // Don't remove ** - we handle it separately
    .replace(/__/g, '')   // Remove alternative bold markers
    .replace(/(?<!\*)\*(?!\*)/g, '') // Remove single * (italic) but keep **
    .replace(/_/g, '')    // Remove alternative italic markers
    .replace(/`/g, '')    // Remove inline code markers
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Convert links to plain text
    .replace(/~~(.+?)~~/g, '$1') // Remove strikethrough
    .trim()
}


// Sanitize filename
function sanitizeFilename(name: string): string {
  return name.replace(/[^\w\-]+/g, '_').substring(0, 50)
}
function renderTextWithInlineBold(
  doc: jsPDF,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  fontSize: number
): number {
  const cleanText = text
    .replace(/`/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .trim()
  
  const parts = cleanText.split(/(\*\*[^*]+\*\*)/g).filter(p => p)
  
  let currentY = y
  let currentLine = ''
  let currentLineSegments: Array<{text: string, bold: boolean}> = []
  
  doc.setFontSize(fontSize)  // Use the passed fontSize
  
  for (const part of parts) {
    if (!part) continue
    
    const isBold = part.startsWith('**') && part.endsWith('**')
    const textContent = isBold ? part.slice(2, -2) : part
    
    const words = textContent.split(' ')
    
    for (let i = 0; i < words.length; i++) {
      const word = words[i]
      const testLine = currentLine + (currentLine ? ' ' : '') + word
      
      doc.setFont('helvetica', isBold ? 'bold' : 'normal')
      const testWidth = doc.getTextWidth(testLine)
      
      if (testWidth > maxWidth && currentLine) {
        renderLine(doc, currentLineSegments, x, currentY)
        currentY += (fontSize * 0.5)  // Line spacing based on font size
        currentLine = word
        currentLineSegments = [{text: word, bold: isBold}]
      } else {
        if (currentLine) currentLine += ' '
        currentLine += word
        
        if (currentLineSegments.length > 0 && 
            currentLineSegments[currentLineSegments.length - 1].bold === isBold) {
          currentLineSegments[currentLineSegments.length - 1].text += 
            (currentLineSegments[currentLineSegments.length - 1].text ? ' ' : '') + word
        } else {
          currentLineSegments.push({text: word, bold: isBold})
        }
      }
    }
  }
  
  if (currentLine) {
    renderLine(doc, currentLineSegments, x, currentY)
    currentY += (fontSize * 0.5)
  }
  return currentY
}

function renderLine(
  doc: jsPDF, 
  segments: Array<{text: string, bold: boolean}>, 
  x: number, 
  y: number
) {
  let currentX = x
  
  for (const segment of segments) {
    doc.setFont('helvetica', segment.bold ? 'bold' : 'normal')
    doc.text(segment.text, currentX, y)
    currentX += doc.getTextWidth(segment.text) + doc.getTextWidth(' ')
  }
}
