const highlights = [
  {
    title: "Fast startup",
    text: "Rsbuild keeps the feedback loop tight with an efficient bundler and modern defaults.",
  },
  {
    title: "JavaScript-first",
    text: "The project uses plain JavaScript, so you can move quickly without TypeScript overhead.",
  },
  {
    title: "React-ready",
    text: "React 19, a clean component structure, and a polished landing page are ready to extend.",
  },
];

export default function App() {
  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">Base website ready</p>
        <p className="eyebrow">React + JavaScript + Rsbuild</p>
        <h1>Build quickly with a modern, efficient starter.</h1>
        <p className="lede">
          This workspace is set up with a lightweight React app that uses Rsbuild instead of Vite.
        </p>

        <div className="cta-row">
          <a className="button primary" href="https://rsbuild.dev" target="_blank" rel="noreferrer">
            Rsbuild docs
          </a>
          <a className="button secondary" href="https://react.dev" target="_blank" rel="noreferrer">
            React docs
          </a>
        </div>
      </section>

      <footer className="footer">
        <span>Starter structure in place.</span>
        <span>Edit <strong>src/App.jsx</strong> to grow the site.</span>
      </footer>

      <section className="panel">
        <div className="panel-header">
          <span>Starter highlights</span>
          <span>Ready to edit</span>
        </div>

        <div className="grid">
          {highlights.map((item) => (
            <article className="card" key={item.title}>
              <h2>{item.title}</h2>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
