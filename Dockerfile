# Builder stage
FROM node:24-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./

# --ignore-scripts: nothing here needs to compile.
#
# better-sqlite3 ships prebuilt binaries for linux-x64 and linux-arm64 in its
# tarball and loads them at require time; sqlite-vec resolves its extension from
# optional per-platform packages. But better-sqlite3's tarball also contains
# binding.gyp, so npm runs `node-gyp rebuild` on install -- and node:24-slim
# ships no Python, so that step died with "Could not find any Python
# installation to use" while every other job stayed green.
#
# Skipping scripts is only safe if the prebuilds are actually used, so the next
# RUN is the falsifier: it loads both native modules and creates a vec0 table,
# failing the build if a prebuild ever goes missing instead of shipping an image
# that dies at runtime.
RUN npm ci --ignore-scripts
RUN node -e "const D=require('better-sqlite3');const db=new D(':memory:');require('sqlite-vec').load(db);db.exec('create virtual table t using vec0(embedding float[4])');console.log('native modules OK');"
COPY . .
RUN npm run build

RUN npm prune --omit=dev

# Production stage
FROM node:24-slim AS production

WORKDIR /app
COPY --from=builder /app/build ./build
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/package-lock.json ./package-lock.json

EXPOSE 3000
ENTRYPOINT [ "node", "build/index.js", "--http", "--port", "3000" ]
