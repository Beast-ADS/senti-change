param([string]$Manifest)
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.SelectVoice("Microsoft Zira Desktop")
$voice.Rate = -1
$voice.Volume = 100
foreach ($item in $items) {
  $voice.SetOutputToWaveFile($item.path)
  $voice.Speak($item.text)
  $voice.SetOutputToNull()
}
$voice.Dispose()
