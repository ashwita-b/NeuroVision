// static/auth.js
document.addEventListener("DOMContentLoaded", () => {
  console.log("Auth JS Loaded");

  // Tab switching functionality
  const tabBtns = document.querySelectorAll(".tab-btn");
  const forms = document.querySelectorAll(".form");
  const switchLinks = document.querySelectorAll(".switch-form");

  // Tab switching
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = btn.getAttribute("href").substring(1);

      // Update active tab
      tabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      // Show target form
      forms.forEach((form) => {
        form.classList.remove("active");
        if (form.id === targetId) {
          form.classList.add("active");
        }
      });
    });
  });

  // Form switching links
  switchLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = link.getAttribute("href").substring(1);

      // Update tabs
      tabBtns.forEach((btn) => {
        btn.classList.remove("active");
        if (btn.getAttribute("href") === "#" + targetId) {
          btn.classList.add("active");
        }
      });

      // Show target form
      forms.forEach((form) => {
        form.classList.remove("active");
        if (form.id === targetId) {
          form.classList.add("active");
        }
      });
    });
  });

  // ---------------- REGISTER FORM HANDLING ----------------
  const registerForm = document.getElementById("register");
  if (registerForm) {
    registerForm.addEventListener("submit", (e) => {
      e.preventDefault();

      const inputs = registerForm.querySelectorAll("input");
      const password = inputs[2].value;
      const confirmPassword = inputs[3].value;

      if (password !== confirmPassword) {
        alert("❌ Passwords do not match");
        return false;
      }

      // For demo purposes only - in real app, this would be handled by Flask
      alert(
        "✅ Registration Successful (Demo)\n\nYou can now login with:\nUsername: admin\nPassword: admin123"
      );

      // Switch to login tab
      document.querySelector('a[href="#login"]').click();

      // Clear form
      registerForm.reset();
    });
  }

  // ---------------- LOGIN FORM ----------------
  // Remove the preventDefault() and let Flask handle the form submission
  // Only keep basic validation
  const loginForm = document.getElementById("login");
  if (loginForm) {
    loginForm.addEventListener("submit", (e) => {
      const email = loginForm
        .querySelector('input[name="username"]')
        .value.trim();
      const password = loginForm
        .querySelector('input[name="password"]')
        .value.trim();

      if (email === "" || password === "") {
        alert("❌ Please fill all fields");
        e.preventDefault();
        return false;
      }

      // Show loading state
      const submitBtn = loginForm.querySelector(".submit-btn");
      const originalText = submitBtn.innerHTML;
      submitBtn.innerHTML =
        '<i class="fas fa-spinner fa-spin"></i> Authenticating...';
      submitBtn.disabled = true;

      // Let the form submit to Flask - Flask will handle the redirect
      console.log("Login form submitting to Flask...");
    });
  }
});
