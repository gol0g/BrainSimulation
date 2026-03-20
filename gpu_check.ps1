$s = (Get-Counter "\GPU Engine(*engtype_3D)\Utilization Percentage" -SampleInterval 1).CounterSamples
$total = ($s | Measure-Object -Property CookedValue -Sum).Sum
Write-Output ([math]::Round($total, 1))
