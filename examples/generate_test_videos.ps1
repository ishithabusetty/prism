param(
    [string]$OutputDir = (Split-Path -Parent $MyInvocation.MyCommand.Path)
)

$ErrorActionPreference = "Stop"

$out = Resolve-Path $OutputDir
$width = 1280
$height = 720
$fps = 24
$quality = 85
$slideDuration = 1

$msoTrue = -1
$msoFalse = 0
$ppLayoutBlank = 12
$ppMediaTaskStatusDone = 3
$ppMediaTaskStatusFailed = 4

function Add-Rect {
    param($Slide, [double]$X, [double]$Y, [double]$W, [double]$H, [int]$Rgb, [double]$Transparency = 0)
    $shape = $Slide.Shapes.AddShape(1, $X, $Y, $W, $H)
    $shape.Fill.ForeColor.RGB = $Rgb
    $shape.Fill.Transparency = $Transparency
    $shape.Line.Visible = $msoFalse
    return $shape
}

function Add-Label {
    param($Slide, [string]$Text, [double]$X, [double]$Y, [double]$W, [double]$H, [int]$Rgb = 16777215, [int]$Size = 24)
    $box = $Slide.Shapes.AddTextbox(1, $X, $Y, $W, $H)
    $box.TextFrame.TextRange.Text = $Text
    $box.TextFrame.TextRange.Font.Color.RGB = $Rgb
    $box.TextFrame.TextRange.Font.Size = $Size
    $box.TextFrame.TextRange.Font.Bold = $msoTrue
    $box.Line.Visible = $msoFalse
    $box.Fill.Visible = $msoFalse
    return $box
}

function Add-Watermark {
    param($Slide, [string]$Mode)
    if ($Mode -eq "tampered") {
        Add-Rect $Slide 960 56 236 72 13882323 0.08 | Out-Null
        Add-Rect $Slide 984 72 188 18 12105912 0.15 | Out-Null
        Add-Rect $Slide 984 100 188 14 13027014 0.2 | Out-Null
        Add-Label $Slide "patched area" 995 82 165 25 5592405 15 | Out-Null
        return
    }
    Add-Rect $Slide 960 56 236 72 16777215 0 | Out-Null
    Add-Rect $Slide 978 76 38 32 5896975 0 | Out-Null
    Add-Rect $Slide 1030 76 38 32 5896975 0 | Out-Null
    Add-Rect $Slide 1082 76 38 32 5896975 0 | Out-Null
    Add-Label $Slide "LICENSED" 972 130 230 28 16777215 16 | Out-Null
}

function Add-Scene {
    param($Slide, [int]$Index, [string]$Kind)

    $colors = @(8013602, 3692793, 10070600, 4708863, 13345587, 5933416, 11079591, 8618883)
    $bg = $colors[$Index % $colors.Length]
    Add-Rect $Slide 0 0 1280 720 $bg | Out-Null

    if ($Kind -eq "plagiarism") {
        Add-Rect $Slide 0 0 1280 720 16777215 0.93 | Out-Null
        Add-Label $Slide "frame-level duplicate: resized / recompressed visual copy" 54 635 820 34 2236962 18 | Out-Null
    }
    elseif ($Kind -eq "reencoded") {
        Add-Rect $Slide 0 0 1280 720 16764057 0.9 | Out-Null
        Add-Label $Slide "re-encoded licensed clip: timestamp drift + color shift" 54 635 820 34 2236962 18 | Out-Null
    }
    elseif ($Kind -eq "tampered") {
        Add-Label $Slide "watermark tamper: localized blur / inpaint patch" 54 635 820 34 16777215 18 | Out-Null
    }
    else {
        Add-Label $Slide "licensed master reference" 54 635 500 34 16777215 18 | Out-Null
    }

    $phase = $Index % 8
    $x = 70 + ($phase * 112)
    $y = 205 + (42 * [Math]::Sin($Index / 1.6))
    Add-Rect $Slide $x $y 168 92 16448250 | Out-Null
    Add-Rect $Slide $x $y 168 14 1973790 | Out-Null
    Add-Rect $Slide (930 - ($phase * 55)) 360 126 126 2634300 | Out-Null
    Add-Rect $Slide 76 72 270 74 16777215 0.08 | Out-Null
    Add-Label $Slide ("SCENE " + ([Math]::Floor($Index / 2) + 1)) 92 87 230 45 16777215 28 | Out-Null

    if ($Kind -eq "tampered") {
        Add-Watermark $Slide "tampered"
    }
    else {
        Add-Watermark $Slide "clean"
    }

    $Slide.SlideShowTransition.AdvanceOnTime = $msoTrue
    $Slide.SlideShowTransition.AdvanceTime = $slideDuration
}

function Export-Fixture {
    param($PowerPoint, [string]$Name, [string]$Kind)

    $presentation = $PowerPoint.Presentations.Add($msoFalse)
    $presentation.PageSetup.SlideWidth = $width
    $presentation.PageSetup.SlideHeight = $height

    for ($i = 0; $i -lt 8; $i++) {
        $slide = $presentation.Slides.Add($i + 1, $ppLayoutBlank)
        Add-Scene $slide $i $Kind
    }

    $pptxPath = Join-Path $out.Path "$Name.pptx"
    $mp4Path = Join-Path $out.Path "$Name.mp4"
    if (Test-Path -LiteralPath $mp4Path) {
        Remove-Item -LiteralPath $mp4Path -Force
    }
    $presentation.SaveAs($pptxPath)
    $presentation.CreateVideo($mp4Path, $msoTrue, $slideDuration, 720, $fps, $quality)

    while ($presentation.CreateVideoStatus -ne $ppMediaTaskStatusDone) {
        if ($presentation.CreateVideoStatus -eq $ppMediaTaskStatusFailed) {
            throw "PowerPoint failed to export $Name.mp4"
        }
        Start-Sleep -Seconds 1
    }

    $presentation.Close()
    Remove-Item -LiteralPath $pptxPath -Force
}

$powerPoint = New-Object -ComObject PowerPoint.Application
try {
    Export-Fixture $powerPoint "licensed_master_clip" "master"
    Export-Fixture $powerPoint "plagiarism_frame_duplication" "plagiarism"
    Export-Fixture $powerPoint "watermark_tamper_blur_inpaint" "tampered"
    Export-Fixture $powerPoint "reencoding_licensed_clip_bypass" "reencoded"
}
finally {
    $powerPoint.Quit()
}

$manifest = [ordered]@{
    generated_with = "PowerPoint CreateVideo H.264 MP4 export"
    fps = $fps
    resolution = "1280x720"
    duration_seconds_each = 8
    watermark_box_pixels = @(960, 56, 236, 72)
    files = @(
        "licensed_master_clip.mp4",
        "plagiarism_frame_duplication.mp4",
        "watermark_tamper_blur_inpaint.mp4",
        "reencoding_licensed_clip_bypass.mp4"
    )
}

$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $out.Path "fixture_manifest.json") -Encoding UTF8
