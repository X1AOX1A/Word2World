<template>
  <div class="not-found">
    <div class="not-found-content">
      <h1>404</h1>
      <div class="error-icon">🤖</div>
      <h2>Page Not Found</h2>
      <p>The environment you're looking for doesn't exist or has been moved.</p>
      <router-link to="/" class="home-btn">
        Go Back Home
      </router-link>
    </div>
  </div>
</template>

<style scoped>
.not-found {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  text-align: center;
}

.not-found-content {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 3rem;
  max-width: 500px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  animation: fadeIn 0.5s ease-out;
}

h1 {
  font-size: 5rem;
  margin: 0;
  font-weight: 800;
  color: rgba(255, 255, 255, 0.8);
  text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.error-icon {
  font-size: 5rem;
  margin: 1rem 0;
  animation: float 3s ease-in-out infinite;
}

h2 {
  font-size: 2rem;
  margin: 0.5rem 0;
  font-weight: 600;
}

p {
  margin: 1rem 0 2rem;
  opacity: 0.9;
  line-height: 1.6;
}

.home-btn {
  display: inline-block;
  background: white;
  color: #764ba2;
  padding: 0.8rem 2rem;
  border-radius: 30px;
  font-weight: 600;
  text-decoration: none;
  transition: all 0.3s;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.home-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-20px); }
}

@media (max-width: 768px) {
  .not-found-content {
    padding: 2rem;
    margin: 0 1rem;
  }
  
  h1 {
    font-size: 4rem;
  }
  
  .error-icon {
    font-size: 4rem;
  }
  
  h2 {
    font-size: 1.5rem;
  }
}
</style> 