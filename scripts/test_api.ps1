# Set UTF-8 encoding for PowerShell
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Function to make API calls with proper encoding
function Test-API {
    param (
        [string]$Endpoint,
        [string]$Method = "GET",
        [object]$Body = $null
    )
    
    $uri = "http://localhost:7860$Endpoint"
    $headers = @{
        "Content-Type" = "application/json; charset=utf-8"
    }
    
    try {
        if ($Method -eq "GET") {
            $response = Invoke-RestMethod -Uri $uri -Method $Method -Headers $headers
        } else {
            $jsonBody = if ($Body) { $Body | ConvertTo-Json -Compress } else { "" }
            $response = Invoke-RestMethod -Uri $uri -Method $Method -Headers $headers -Body $jsonBody
        }
        
        # Handle the response based on its type
        if ($response -is [string]) {
            Write-Host $response
        } else {
            $response | ConvertTo-Json -Depth 10
        }
    }
    catch {
        Write-Host "Error: $_"
    }
}

# Test health check
Write-Host "`nTesting health check endpoint..."
Test-API -Endpoint "/"

# Test model info
Write-Host "`nTesting model info endpoint..."
Test-API -Endpoint "/info"

# Test query endpoint
Write-Host "`nTesting query endpoint..."
$queryBody = @{
    text = "请介绍一下溪水旁杂志"
    language = "simplified"
}
Test-API -Endpoint "/query" -Method "POST" -Body $queryBody 