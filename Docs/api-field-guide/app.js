const root = document.documentElement;
const button = document.querySelector("#theme-toggle");

button.addEventListener("click", () => {
  const dark = root.dataset.theme !== "dark";
  root.dataset.theme = dark ? "dark" : "light";
  button.textContent = dark ? "Light mode" : "Dark mode";
});

const links = [...document.querySelectorAll("nav a")];
const sections = [...document.querySelectorAll("main section")];

const observer = new IntersectionObserver((entries) => {
  const visible = entries.find((entry) => entry.isIntersecting);
  if (!visible) return;
  links.forEach((link) => link.classList.toggle("active", link.hash === `#${visible.target.id}`));
}, { rootMargin: "-30% 0px -60%" });

sections.forEach((section) => observer.observe(section));

links.forEach(link => link.addEventListener('click', () => {
  document.querySelector('#section-label').textContent = link.textContent.trim().replace(/\s+/, ' / ');
}));
document.querySelectorAll('.copy').forEach(copy => {
  copy.addEventListener('click', async () => {
    const text = copy.closest('.example').querySelector('pre code').textContent;
    try {
      await navigator.clipboard.writeText(text);
      copy.textContent = 'Copied';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(copy.closest('.example').querySelector('pre code'));
      const selection = window.getSelection();
      selection.removeAllRanges(); selection.addRange(range);
      copy.textContent = 'Selected — copy manually';
    }
    setTimeout(() => { copy.textContent = 'Copy'; }, 2200);
  });
});
