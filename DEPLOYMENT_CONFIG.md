# Database Configuration Guide for Multiple Cloud Providers

This guide explains how to configure the Konkani Dictionary application to work with different cloud database providers (Railway, Google Cloud SQL, Azure Database) while maintaining local development flexibility.

## Overview

The application uses a provider-agnostic database configuration that automatically detects and adapts to different PostgreSQL providers based on environment variables.

## Configuration Priority

1. **DATABASE_URL** (highest priority) - Full connection string from cloud provider
2. **Individual PG* variables** - For custom setups or local development
3. **Defaults** - Fallback values for local development

## Supported Providers

### Railway (Current Production)

**Environment Variables:**
```bash
DATABASE_URL=postgresql://postgres:password@containers-us-west-1.railway.app:1234/railway
NODE_ENV=production
```

**SSL Configuration:**
- Automatically configured for Railway's self-signed certificates
- `rejectUnauthorized: false` by default

### Google Cloud SQL

**Environment Variables:**
```bash
DATABASE_URL=postgresql://user:password@127.0.0.1:5432/database?host=/cloudsql/project:region:instance
# OR individual variables:
PGHOST=/cloudsql/project:region:instance
PGUSER=your-user
PGPASSWORD=your-password
PGDATABASE=your-database
PGPORT=5432

# SSL (if required)
DB_SSL_CA=/path/to/ca-cert.pem
DB_SSL_CERT=/path/to/client-cert.pem
DB_SSL_KEY=/path/to/client-key.pem
```

**Connection Notes:**
- Uses Cloud SQL Proxy for secure connections
- May require SSL certificates for authentication

### Azure Database for PostgreSQL

**Environment Variables:**
```bash
DATABASE_URL=postgresql://user@server:password@server.postgres.database.azure.com:5432/database
# OR individual variables:
PGHOST=your-server.postgres.database.azure.com
PGUSER=user@server
PGPASSWORD=your-password
PGDATABASE=your-database
PGPORT=5432

# SSL (Azure requires SSL)
DB_SSL_REJECT_UNAUTHORIZED=false
```

**Connection Notes:**
- Requires SSL connection (enforced by Azure)
- Username format: `user@server`

### AWS RDS PostgreSQL

**Environment Variables:**
```bash
DATABASE_URL=postgresql://user:password@your-instance.rds.amazonaws.com:5432/database
# OR individual variables:
PGHOST=your-instance.rds.amazonaws.com
PGUSER=your-user
PGPASSWORD=your-password
PGDATABASE=your-database
PGPORT=5432

# SSL (RDS supports SSL)
DB_SSL_REJECT_UNAUTHORIZED=false
DB_SSL_CA=/path/to/rds-ca-cert.pem
```

### Local PostgreSQL (Development)

**Environment Variables:**
```bash
PGHOST=localhost
PGUSER=konkani_dev
PGPASSWORD=your-local-password
PGDATABASE=konkani_dictionary
PGPORT=5432
NODE_ENV=development
```

**No SSL required for local development.**

## Connection Pool Configuration

Adjust these based on your provider's limits and application needs:

```bash
# Default values (Railway-optimized)
DB_POOL_MAX=20              # Maximum connections
DB_IDLE_TIMEOUT=30000       # Close idle connections after 30s
DB_CONNECTION_TIMEOUT=2000  # Connection timeout 2s
```

**Provider Recommendations:**
- **Railway:** Default values work well
- **Google Cloud SQL:** May need higher timeouts
- **Azure:** Check connection limits in your tier
- **AWS RDS:** Adjust based on instance size

## SSL Configuration Options

```bash
# Basic SSL control
DB_SSL_REJECT_UNAUTHORIZED=false  # Allow self-signed certs (Railway, some cloud providers)

# Certificate paths (for providers requiring client certificates)
DB_SSL_CA=/path/to/ca-certificate.pem
DB_SSL_CERT=/path/to/client-certificate.pem
DB_SSL_KEY=/path/to/client-key.pem
```

## Environment-Specific Configuration Files

Create `.env` files for different environments:

### `.env.railway`
```bash
DATABASE_URL=postgresql://postgres:password@containers-us-west-1.railway.app:1234/railway
NODE_ENV=production
API_BASE_URL=https://your-railway-app.railway.app/api
```

### `.env.gcp`
```bash
DATABASE_URL=postgresql://user:password@127.0.0.1:5432/database?host=/cloudsql/project:region:instance
NODE_ENV=production
DB_SSL_CA=/secrets/cloudsql-ca.pem
API_BASE_URL=https://your-gcp-app.appspot.com/api
```

### `.env.azure`
```bash
PGHOST=your-server.postgres.database.azure.com
PGUSER=user@server
PGPASSWORD=password
PGDATABASE=database
PGPORT=5432
NODE_ENV=production
DB_SSL_REJECT_UNAUTHORIZED=false
API_BASE_URL=https://your-azure-app.azurewebsites.net/api
```

### `.env.local`
```bash
PGHOST=localhost
PGUSER=konkani_dev
PGPASSWORD=dev-password
PGDATABASE=konkani_dictionary
PGPORT=5432
NODE_ENV=development
API_BASE_URL=http://localhost:3002/api
```

## Frontend Configuration

The frontend automatically detects the environment:

- **Local Development:** `http://localhost:3002/api`
- **Production:** `https://your-domain.com/api`

Override with environment variable:
```bash
API_BASE_URL=https://custom-api-endpoint.com/api
```

## Debugging Database Connections

Use the debug endpoint to verify configuration:

```bash
curl https://your-app.com/api/debug/db
```

Response includes:
- Database provider detection
- Connection status
- SSL configuration
- Pool settings
- Connection details (without passwords)

## Migration Safety

Database migrations are protected in production:

- **Development:** Migrations run automatically
- **Production:** Requires explicit flags:
  ```bash
  MIGRATE_ALLOW=true
  MIGRATE_SECRET=your-secret-key
  ```

  And header:
  ```bash
  curl -X POST https://your-app.com/api/migrate \
    -H "x-migrate-secret: your-secret-key"
  ```

## Troubleshooting

### Connection Issues

1. **Check debug endpoint:** `/api/debug/db`
2. **Verify environment variables** are loaded correctly
3. **Test connection string** format for your provider
4. **Check SSL requirements** for your cloud provider

### Common Issues

- **Railway:** Ensure `rejectUnauthorized: false` for SSL
- **Google Cloud SQL:** Verify Cloud SQL Proxy is running
- **Azure:** Confirm username format (`user@server`)
- **AWS RDS:** Check security group and VPC settings

### Performance Tuning

- **Connection Pool:** Adjust `DB_POOL_MAX` based on provider limits
- **Timeouts:** Increase for high-latency connections (Google Cloud)
- **SSL:** Disable verification only when necessary (development/local)

## Provider-Specific Notes

### Railway
- Uses self-signed certificates
- Connection pooling works well with defaults
- Automatic SSL configuration

### Google Cloud SQL
- Requires Cloud SQL Proxy or direct IP access
- May need SSL certificates
- Higher latency possible

### Azure Database
- Enforces SSL connections
- Username includes server name
- Connection limits vary by pricing tier

### AWS RDS
- SSL optional but recommended
- Use RDS CA certificate for SSL
- Connection limits based on instance type

This configuration ensures the application can seamlessly switch between cloud providers with minimal code changes, only requiring environment variable updates.