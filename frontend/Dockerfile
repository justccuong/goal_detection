FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy application files
COPY . .

# Expose Vite default port
EXPOSE 5173

# Start the Vite development server in host mode
CMD ["npm", "run", "dev", "--", "--host"]
