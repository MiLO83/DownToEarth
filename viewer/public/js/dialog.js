// Simple dialog popup. Replaceable with anything richer later.

export class Dialog {
  constructor(root) {
    this.root = root;
    this.speakerEl = root.querySelector('.speaker');
    this.bodyEl = root.querySelector('.body');
    this._timeout = null;
    root.addEventListener('click', () => this.hide());
  }
  show(speaker, body, autoHideMs = 7000) {
    this.speakerEl.textContent = speaker;
    this.bodyEl.textContent = body;
    this.root.classList.add('visible');
    clearTimeout(this._timeout);
    if (autoHideMs > 0) this._timeout = setTimeout(() => this.hide(), autoHideMs);
  }
  hide() {
    this.root.classList.remove('visible');
    clearTimeout(this._timeout);
  }
}
