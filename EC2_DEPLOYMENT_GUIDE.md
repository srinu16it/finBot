# EC2 Deployment Guide for FinBot

## Overview

When deploying FinBot on AWS EC2, there are special considerations for data providers, particularly Yahoo Finance which may rate limit or block requests from cloud provider IP addresses.

## Known Issues

### Yahoo Finance on EC2

Yahoo Finance actively blocks or heavily rate limits requests from AWS EC2 instances. This is because:
- Cloud provider IPs are often used for automated scraping
- Yahoo Finance enforces stricter limits on these IP ranges
- The error typically appears as `429 Client Error: Too Many Requests`

## Solutions

### 1. Use AlphaVantage as Primary Provider (Recommended)

The system automatically detects when running on EC2 and prefers AlphaVantage if available:

```bash
# Set your AlphaVantage API key
export ALPHAVANTAGE_API_KEY="your-api-key-here"

# The system will automatically use AlphaVantage on EC2
```

### 2. Force a Specific Provider

You can override the automatic provider selection:

```bash
# Always use AlphaVantage
export DATA_PROVIDER_PREFERENCE="alphavantage"

# Always use Yahoo (not recommended on EC2)
export DATA_PROVIDER_PREFERENCE="yahoo"

# Let the system decide (default)
export DATA_PROVIDER_PREFERENCE="auto"
```

### 3. Adjust Rate Limiting

If you still want to try Yahoo Finance on EC2, you can adjust the rate limiting:

```bash
# Increase delay between Yahoo requests on EC2 (default: 5 seconds)
export YAHOO_EC2_RATE_LIMIT="10.0"

# For comparison, local rate limit (default: 2 seconds)
export YAHOO_LOCAL_RATE_LIMIT="2.0"
```

### 4. Enable Automatic Fallback

The system can automatically fallback to AlphaVantage when Yahoo fails:

```bash
# Enable automatic fallback (default: true)
export AUTO_FALLBACK_TO_ALPHAVANTAGE="true"
```

## Complete EC2 Setup Example

1. **Install dependencies:**
```bash
# Update system
sudo yum update -y  # For Amazon Linux
# or
sudo apt-get update  # For Ubuntu

# Install Python 3.9+
sudo yum install python3.9 python3.9-pip -y

# Clone repository
git clone https://github.com/your-repo/finbot.git
cd finbot

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
# Create .env file
cat > .env << EOF
# AlphaVantage API key (required for EC2)
ALPHAVANTAGE_API_KEY=your-api-key-here

# Force AlphaVantage on EC2
DATA_PROVIDER_PREFERENCE=alphavantage

# Increase rate limits for safety
YAHOO_EC2_RATE_LIMIT=10.0

# Enable automatic fallback
AUTO_FALLBACK_TO_ALPHAVANTAGE=true
EOF
```

3. **Run the application:**
```bash
# For Streamlit UI
streamlit run enhancements/examples/pattern_analysis_streamlit.py --server.port 8501 --server.address 0.0.0.0

# Make sure to open port 8501 in your EC2 security group
```

## Security Group Configuration

Add the following inbound rules to your EC2 security group:
- Type: Custom TCP
- Port: 8501
- Source: Your IP address (for security) or 0.0.0.0/0 (for public access)

## Monitoring

Check the logs for provider selection:
```
2025-06-01 16:00:00 - INFO - Running on EC2 - preferring AlphaVantage over Yahoo Finance
2025-06-01 16:00:00 - INFO - Using AlphaVantage data provider
```

If Yahoo fails, you'll see:
```
2025-06-01 16:00:00 - WARNING - Yahoo Finance blocked on EC2, falling back to AlphaVantage
```

## Best Practices

1. **Always use AlphaVantage on EC2** - It's more reliable for cloud deployments
2. **Cache data aggressively** - Reduces API calls and improves performance
3. **Monitor rate limits** - Even AlphaVantage has limits (5/min for free tier)
4. **Use environment variables** - Don't hardcode API keys

## Troubleshooting

### Issue: Yahoo Finance returns 429 errors
**Solution:** Use AlphaVantage or increase `YAHOO_EC2_RATE_LIMIT`

### Issue: Analysis fails with "No data"
**Solution:** Check that AlphaVantage API key is set correctly

### Issue: Slow performance
**Solution:** The extended rate limiting on EC2 makes Yahoo slower. Use AlphaVantage.

### Issue: Can't access Streamlit
**Solution:** Check security group allows inbound traffic on port 8501 