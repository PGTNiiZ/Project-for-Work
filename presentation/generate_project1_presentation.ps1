param(
    [string]$DataPath = (Join-Path $PSScriptRoot 'slides_data.ps1'),
    [string]$OutputGuide = (Join-Path $PSScriptRoot 'project1_presentation_100slides_th.md'),
    [string]$OutputPptx = (Join-Path $PSScriptRoot 'project1_presentation_100slides_th.pptx')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Convert-ToXmlSafe {
    param([string]$Text)
    if ([string]::IsNullOrEmpty($Text)) {
        return ''
    }
    return [System.Security.SecurityElement]::Escape($Text)
}

function Write-Utf8File {
    param(
        [string]$Path,
        [string]$Content
    )
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $encoding = New-Object System.Text.UTF8Encoding($true)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function New-RectShapeXml {
    param(
        [int]$Id,
        [string]$Name,
        [int]$X,
        [int]$Y,
        [int]$Cx,
        [int]$Cy,
        [string]$FillColor,
        [string]$LineColor = ''
    )

    $lineXml = if ($LineColor) {
        "<a:ln w=`"12700`"><a:solidFill><a:srgbClr val=`"$LineColor`"/></a:solidFill></a:ln>"
    }
    else {
        '<a:ln><a:noFill/></a:ln>'
    }

    @"
<p:sp>
  <p:nvSpPr>
    <p:cNvPr id="$Id" name="$(Convert-ToXmlSafe $Name)"/>
    <p:cNvSpPr/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm>
      <a:off x="$X" y="$Y"/>
      <a:ext cx="$Cx" cy="$Cy"/>
    </a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    <a:solidFill><a:srgbClr val="$FillColor"/></a:solidFill>
    $lineXml
  </p:spPr>
</p:sp>
"@
}

function New-TextBoxXml {
    param(
        [int]$Id,
        [string]$Name,
        [int]$X,
        [int]$Y,
        [int]$Cx,
        [int]$Cy,
        [string[]]$Lines,
        [int]$FontSize = 2000,
        [string]$Color = '0F172A',
        [bool]$Bold = $false,
        [string]$Align = 'l',
        [string]$FillColor = '',
        [string]$LineColor = '',
        [int]$Inset = 91440
    )

    $boldAttr = if ($Bold) { ' b="1"' } else { '' }
    $fillXml = if ($FillColor) {
        "<a:solidFill><a:srgbClr val=`"$FillColor`"/></a:solidFill>"
    }
    else {
        '<a:noFill/>'
    }
    $lineXml = if ($LineColor) {
        "<a:ln w=`"12700`"><a:solidFill><a:srgbClr val=`"$LineColor`"/></a:solidFill></a:ln>"
    }
    else {
        '<a:ln><a:noFill/></a:ln>'
    }

    $paragraphs = foreach ($line in $Lines) {
        $safe = Convert-ToXmlSafe $line
@"
    <a:p>
      <a:pPr algn="$Align"/>
      <a:r>
        <a:rPr lang="th-TH" sz="$FontSize"$boldAttr>
          <a:solidFill><a:srgbClr val="$Color"/></a:solidFill>
        </a:rPr>
        <a:t>$safe</a:t>
      </a:r>
      <a:endParaRPr lang="th-TH" sz="$FontSize">
        <a:solidFill><a:srgbClr val="$Color"/></a:solidFill>
      </a:endParaRPr>
    </a:p>
"@
    }

    @"
<p:sp>
  <p:nvSpPr>
    <p:cNvPr id="$Id" name="$(Convert-ToXmlSafe $Name)"/>
    <p:cNvSpPr txBox="1"/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm>
      <a:off x="$X" y="$Y"/>
      <a:ext cx="$Cx" cy="$Cy"/>
    </a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    $fillXml
    $lineXml
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" lIns="$Inset" tIns="$Inset" rIns="$Inset" bIns="$Inset" anchor="t"/>
    <a:lstStyle/>
$($paragraphs -join "`n")
  </p:txBody>
</p:sp>
"@
}

function New-SlideXml {
    param([pscustomobject]$Slide)

    $slideWidth = 12192000
    $slideHeight = 6858000

    $spTreeHeader = @'
<p:spTree>
  <p:nvGrpSpPr>
    <p:cNvPr id="1" name=""/>
    <p:cNvGrpSpPr/>
    <p:nvPr/>
  </p:nvGrpSpPr>
  <p:grpSpPr>
    <a:xfrm>
      <a:off x="0" y="0"/>
      <a:ext cx="0" cy="0"/>
      <a:chOff x="0" y="0"/>
      <a:chExt cx="0" cy="0"/>
    </a:xfrm>
  </p:grpSpPr>
'@

    $spTreeFooter = @'
</p:spTree>
'@

    switch ($Slide.Type) {
        'title' {
            $shapes = @(
                (New-RectShapeXml -Id 2 -Name 'Background' -X 0 -Y 0 -Cx $slideWidth -Cy $slideHeight -FillColor 'F8FAFC')
                (New-RectShapeXml -Id 3 -Name 'Accent Bar' -X 0 -Y 0 -Cx $slideWidth -Cy 457200 -FillColor '0F766E')
                (New-TextBoxXml -Id 4 -Name 'Title' -X 685800 -Y 1028700 -Cx 10972800 -Cy 914400 -Lines @($Slide.Title) -FontSize 3200 -Color '0F172A' -Bold $true)
                (New-TextBoxXml -Id 5 -Name 'Subtitle' -X 685800 -Y 2057400 -Cx 8500000 -Cy 1828800 -Lines $Slide.Bullets -FontSize 2000 -Color '334155')
                (New-TextBoxXml -Id 6 -Name 'Visual Prompt' -X 685800 -Y 5029200 -Cx 10515600 -Cy 685800 -Lines @('แนวภาพหน้าปก: ' + $Slide.Visual) -FontSize 1400 -Color '64748B' -FillColor 'E2E8F0' -LineColor 'CBD5E1')
            )
        }
        'divider' {
            $subtitle = if ($Slide.Bullets.Count -gt 0) { $Slide.Bullets[0] } else { '' }
            $shapes = @(
                (New-RectShapeXml -Id 2 -Name 'Background' -X 0 -Y 0 -Cx $slideWidth -Cy $slideHeight -FillColor '0F766E')
                (New-TextBoxXml -Id 3 -Name 'Section Label' -X 914400 -Y 1219200 -Cx 5000000 -Cy 457200 -Lines @($Slide.Section) -FontSize 1600 -Color 'CCFBF1' -Bold $true)
                (New-TextBoxXml -Id 4 -Name 'Section Title' -X 914400 -Y 1828800 -Cx 10363200 -Cy 1143000 -Lines @($Slide.Title) -FontSize 3400 -Color 'FFFFFF' -Bold $true)
                (New-TextBoxXml -Id 5 -Name 'Section Subtitle' -X 914400 -Y 3200400 -Cx 9000000 -Cy 914400 -Lines @($subtitle) -FontSize 1800 -Color 'E0F2FE')
            )
        }
        'closing' {
            $shapes = @(
                (New-RectShapeXml -Id 2 -Name 'Background' -X 0 -Y 0 -Cx $slideWidth -Cy $slideHeight -FillColor '0F172A')
                (New-TextBoxXml -Id 3 -Name 'Title' -X 914400 -Y 1371600 -Cx 10363200 -Cy 914400 -Lines @($Slide.Title) -FontSize 3400 -Color 'FFFFFF' -Bold $true -Align 'ctr')
                (New-TextBoxXml -Id 4 -Name 'Bullets' -X 1828800 -Y 2514600 -Cx 8534400 -Cy 2057400 -Lines $Slide.Bullets -FontSize 2000 -Color 'CBD5E1' -Align 'ctr')
                (New-TextBoxXml -Id 5 -Name 'Visual Prompt' -X 1828800 -Y 4800600 -Cx 8534400 -Cy 685800 -Lines @('ภาพปิดท้าย: ' + $Slide.Visual) -FontSize 1400 -Color '94A3B8' -Align 'ctr')
            )
        }
        default {
            $bulletLines = @()
            foreach ($item in $Slide.Bullets) {
                if ($item) {
                    $bulletLines += ('• ' + $item)
                }
            }

            $visualLines = @('ภาพประกอบแนะนำ', $Slide.Visual)

            $shapes = @(
                (New-RectShapeXml -Id 2 -Name 'Top Bar' -X 0 -Y 0 -Cx $slideWidth -Cy 365760 -FillColor '0F766E')
                (New-TextBoxXml -Id 3 -Name 'Title' -X 685800 -Y 457200 -Cx 7315200 -Cy 685800 -Lines @($Slide.Title) -FontSize 2600 -Color '0F172A' -Bold $true)
                (New-TextBoxXml -Id 4 -Name 'Bullets' -X 685800 -Y 1188720 -Cx 6705600 -Cy 3886200 -Lines $bulletLines -FontSize 1900 -Color '1E293B')
                (New-TextBoxXml -Id 5 -Name 'Visual Box' -X 7924800 -Y 1280160 -Cx 3566160 -Cy 2697480 -Lines $visualLines -FontSize 1500 -Color '334155' -Bold $false -FillColor 'F8FAFC' -LineColor 'CBD5E1')
                (New-TextBoxXml -Id 6 -Name 'Footer Left' -X 685800 -Y 6126480 -Cx 5000000 -Cy 228600 -Lines @($Slide.Section) -FontSize 1200 -Color '64748B')
                (New-TextBoxXml -Id 7 -Name 'Footer Right' -X 9144000 -Y 6126480 -Cx 2340000 -Cy 228600 -Lines @("Slide $($Slide.Number)") -FontSize 1200 -Color '64748B' -Align 'r')
            )
        }
    }

@"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="$(Convert-ToXmlSafe $Slide.Title)">
$spTreeHeader
$($shapes -join "`n")
$spTreeFooter
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sld>
"@
}

function New-MarkdownGuide {
    param([object[]]$Slides)

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('# Project 1 Presentation Guide')
    $lines.Add('')
    $lines.Add('เอกสารนี้เป็นแม่แบบนำเสนอประมาณ 100 สไลด์สำหรับ Project 1 โดยแยก 4 ส่วนให้ชัดเจนคือ ข้อความที่ควรอยู่บนสไลด์, แนวภาพประกอบ, ไฟล์ภาพใน repo ที่หยิบใช้ได้ และสคริปต์พูดสำหรับซ้อมนำเสนอ')
    $lines.Add('')

    foreach ($slide in $Slides) {
        $lines.Add("## Slide $($slide.Number): $($slide.Title)")
        $lines.Add('')
        $lines.Add("- ประเภท: $($slide.Type)")
        $lines.Add("- หมวด: $($slide.Section)")
        $lines.Add('- ข้อความบนสไลด์:')
        foreach ($bullet in $slide.Bullets) {
            $lines.Add("- $bullet")
        }
        $lines.Add("- ภาพประกอบที่ควรใช้: $($slide.Visual)")
        if ($slide.Asset) {
            $lines.Add("- ไฟล์ใน repo ที่ใช้ได้: $($slide.Asset)")
        }
        $lines.Add('')
        $lines.Add('**Speaker Script**')
        $lines.Add($slide.Script)
        $lines.Add('')
    }

    return ($lines -join "`r`n")
}

function New-ThemeXml {
@'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Project 1 Theme">
  <a:themeElements>
    <a:clrScheme name="Project Colors">
      <a:dk1><a:srgbClr val="0F172A"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="134E4A"/></a:dk2>
      <a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="0F766E"/></a:accent1>
      <a:accent2><a:srgbClr val="0891B2"/></a:accent2>
      <a:accent3><a:srgbClr val="F59E0B"/></a:accent3>
      <a:accent4><a:srgbClr val="22C55E"/></a:accent4>
      <a:accent5><a:srgbClr val="64748B"/></a:accent5>
      <a:accent6><a:srgbClr val="CBD5E1"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink>
      <a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Project Fonts">
      <a:majorFont>
        <a:latin typeface="Aptos Display"/>
        <a:ea typeface="Angsana New"/>
        <a:cs typeface="Aptos"/>
      </a:majorFont>
      <a:minorFont>
        <a:latin typeface="Aptos"/>
        <a:ea typeface="Angsana New"/>
        <a:cs typeface="Aptos"/>
      </a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Project Format">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"><a:tint val="50000"/><a:satMod val="300000"/></a:schemeClr></a:solidFill>
        <a:gradFill rotWithShape="1">
          <a:gsLst>
            <a:gs pos="0"><a:schemeClr val="phClr"><a:tint val="50000"/><a:satMod val="300000"/></a:schemeClr></a:gs>
            <a:gs pos="35000"><a:schemeClr val="phClr"><a:tint val="37000"/><a:satMod val="300000"/></a:schemeClr></a:gs>
            <a:gs pos="100000"><a:schemeClr val="phClr"><a:tint val="15000"/><a:satMod val="350000"/></a:schemeClr></a:gs>
          </a:gsLst>
          <a:lin ang="16200000" scaled="1"/>
        </a:gradFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525" cap="flat" cmpd="sng" algn="ctr">
          <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
          <a:prstDash val="solid"/>
        </a:ln>
        <a:ln w="25400" cap="flat" cmpd="sng" algn="ctr">
          <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
          <a:prstDash val="solid"/>
        </a:ln>
        <a:ln w="38100" cap="flat" cmpd="sng" algn="ctr">
          <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
          <a:prstDash val="solid"/>
        </a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
      </a:effectStyleLst>
      <a:bgFillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"><a:tint val="95000"/><a:satMod val="170000"/></a:schemeClr></a:solidFill>
        <a:gradFill rotWithShape="1">
          <a:gsLst>
            <a:gs pos="0"><a:schemeClr val="phClr"><a:tint val="93000"/><a:satMod val="150000"/></a:schemeClr></a:gs>
            <a:gs pos="50000"><a:schemeClr val="phClr"><a:tint val="98000"/><a:satMod val="130000"/></a:schemeClr></a:gs>
            <a:gs pos="100000"><a:schemeClr val="phClr"><a:tint val="90000"/><a:satMod val="120000"/></a:schemeClr></a:gs>
          </a:gsLst>
          <a:path path="circle">
            <a:fillToRect l="50000" t="-80000" r="50000" b="180000"/>
          </a:path>
        </a:gradFill>
      </a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>
'@
}

if (-not (Test-Path $DataPath)) {
    throw "Slide data file not found: $DataPath"
}

$slides = & $DataPath
if (-not $slides -or $slides.Count -eq 0) {
    throw 'No slide data was returned from slides_data.ps1'
}

$guideContent = New-MarkdownGuide -Slides $slides
Write-Utf8File -Path $OutputGuide -Content $guideContent

$tempRoot = Join-Path $PSScriptRoot ("_tmp_pptx_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

try {
    $dirs = @(
        '_rels',
        'docProps',
        'ppt',
        'ppt/_rels',
        'ppt/slides',
        'ppt/slides/_rels',
        'ppt/slideLayouts',
        'ppt/slideLayouts/_rels',
        'ppt/slideMasters',
        'ppt/slideMasters/_rels',
        'ppt/theme'
    )
    foreach ($dir in $dirs) {
        New-Item -ItemType Directory -Path (Join-Path $tempRoot $dir) -Force | Out-Null
    }

    $dateIso = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

    $rootRels = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
'@
    Write-Utf8File -Path (Join-Path $tempRoot '_rels/.rels') -Content $rootRels

    $appXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office PowerPoint</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>$($slides.Count)</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Theme</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>1</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="1" baseType="lpstr">
      <vt:lpstr>Project 1 Theme</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
"@
    Write-Utf8File -Path (Join-Path $tempRoot 'docProps/app.xml') -Content $appXml

    $coreXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Project 1 Presentation Guide</dc:title>
  <dc:subject>Cross-Platform Identity Resolution for CRM</dc:subject>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:keywords>presentation,pptx,crm,identity resolution,multimodal</cp:keywords>
  <dc:description>100-slide presentation template with speaking script for Project 1.</dc:description>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">$dateIso</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">$dateIso</dcterms:modified>
</cp:coreProperties>
"@
    Write-Utf8File -Path (Join-Path $tempRoot 'docProps/core.xml') -Content $coreXml

    $slideMasterXml = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="Slide Master">
    <p:bg>
      <p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="2147483649" r:id="rId1"/>
  </p:sldLayoutIdLst>
  <p:txStyles>
    <p:titleStyle>
      <a:lvl1pPr algn="l"><a:defRPr sz="3200" b="1"/></a:lvl1pPr>
    </p:titleStyle>
    <p:bodyStyle>
      <a:lvl1pPr marL="342900" indent="-342900"><a:defRPr sz="2000"/></a:lvl1pPr>
    </p:bodyStyle>
    <p:otherStyle>
      <a:lvl1pPr><a:defRPr sz="1800"/></a:lvl1pPr>
    </p:otherStyle>
  </p:txStyles>
</p:sldMaster>
'@
    Write-Utf8File -Path (Join-Path $tempRoot 'ppt/slideMasters/slideMaster1.xml') -Content $slideMasterXml

    $slideMasterRels = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
'@
    Write-Utf8File -Path (Join-Path $tempRoot 'ppt/slideMasters/_rels/slideMaster1.xml.rels') -Content $slideMasterRels

    $slideLayoutXml = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
'@
    Write-Utf8File -Path (Join-Path $tempRoot 'ppt/slideLayouts/slideLayout1.xml') -Content $slideLayoutXml

    $slideLayoutRels = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
'@
    Write-Utf8File -Path (Join-Path $tempRoot 'ppt/slideLayouts/_rels/slideLayout1.xml.rels') -Content $slideLayoutRels

    Write-Utf8File -Path (Join-Path $tempRoot 'ppt/theme/theme1.xml') -Content (New-ThemeXml)

    $slideRelXml = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>
'@

    $slideOverrides = New-Object System.Collections.Generic.List[string]
    $presentationSlideEntries = New-Object System.Collections.Generic.List[string]
    $presentationRelEntries = New-Object System.Collections.Generic.List[string]

    $presentationRelEntries.Add('<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')

    for ($i = 0; $i -lt $slides.Count; $i++) {
        $slideIndex = $i + 1
        $slidePath = Join-Path $tempRoot ("ppt/slides/slide{0}.xml" -f $slideIndex)
        $slideRelPath = Join-Path $tempRoot ("ppt/slides/_rels/slide{0}.xml.rels" -f $slideIndex)
        Write-Utf8File -Path $slidePath -Content (New-SlideXml -Slide $slides[$i])
        Write-Utf8File -Path $slideRelPath -Content $slideRelXml

        $slideOverrides.Add("<Override PartName=`"/ppt/slides/slide$slideIndex.xml`" ContentType=`"application/vnd.openxmlformats-officedocument.presentationml.slide+xml`"/>")
        $presentationSlideEntries.Add("<p:sldId id=`"$(256 + $i)`" r:id=`"rId$($slideIndex + 1)`"/>")
        $presentationRelEntries.Add("<Relationship Id=`"rId$($slideIndex + 1)`" Type=`"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide`" Target=`"slides/slide$slideIndex.xml`"/>")
    }

    $presentationXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" saveSubsetFonts="1" autoCompressPictures="1">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
$($presentationSlideEntries -join "`n")
  </p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>
"@
    Write-Utf8File -Path (Join-Path $tempRoot 'ppt/presentation.xml') -Content $presentationXml

    $presentationRelsXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
$($presentationRelEntries -join "`n")
</Relationships>
"@
    Write-Utf8File -Path (Join-Path $tempRoot 'ppt/_rels/presentation.xml.rels') -Content $presentationRelsXml

    $contentTypesXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
$($slideOverrides -join "`n")
</Types>
"@
    Write-Utf8File -Path (Join-Path $tempRoot '[Content_Types].xml') -Content $contentTypesXml

    $zipPath = [System.IO.Path]::ChangeExtension($OutputPptx, '.zip')
    if (Test-Path $OutputPptx) {
        Remove-Item -LiteralPath $OutputPptx -Force
    }
    if (Test-Path $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zipFs = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::Create)
    try {
        $archive = New-Object System.IO.Compression.ZipArchive($zipFs, [System.IO.Compression.ZipArchiveMode]::Create, $false)
        try {
            Get-ChildItem -Path $tempRoot -File -Recurse | ForEach-Object {
                $relative = $_.FullName.Substring($tempRoot.Length).TrimStart('\', '/')
                $entryName = $relative -replace '\\', '/'
                $entry = $archive.CreateEntry($entryName, [System.IO.Compression.CompressionLevel]::Optimal)
                $entryStream = $entry.Open()
                $fileStream = [System.IO.File]::OpenRead($_.FullName)
                try {
                    $fileStream.CopyTo($entryStream)
                }
                finally {
                    $fileStream.Dispose()
                    $entryStream.Dispose()
                }
            }
        }
        finally {
            $archive.Dispose()
        }
    }
    finally {
        $zipFs.Dispose()
    }
    Move-Item -LiteralPath $zipPath -Destination $OutputPptx -Force
}
finally {
    if (Test-Path $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

Write-Host "Guide written to: $OutputGuide"
Write-Host "PPTX written to:  $OutputPptx"
