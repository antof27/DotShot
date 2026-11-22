(function () {
  function katexPlugin(hook) {
    let mathBlocks = [];

    hook.beforeEach(function (content) {
      mathBlocks = [];
      const codeBlocks = [];

      content = content.replace(/```[\s\S]*?```/g, function (match) {
        const placeholder = `<!--CODE_BLOCK_PLACEHOLDER_${codeBlocks.length}-->`;
        codeBlocks.push(match);
        return placeholder;
      });

      content = content.replace(/`[^`\n]*?`/g, function (match) {
        const placeholder = `<!--CODE_BLOCK_PLACEHOLDER_${codeBlocks.length}-->`;
        codeBlocks.push(match);
        return placeholder;
      });

      content = content.replace(/\$\$([\s\S]*?)\$\$/g, function (_m, math) {
        const index = mathBlocks.length;
        mathBlocks.push({ math: math.trim(), display: true });
        return `<span class="katex-placeholder" data-index="${index}"></span>`;
      });

      content = content.replace(/\$([^\$\n]+?)\$/g, function (_m, math) {
        const index = mathBlocks.length;
        mathBlocks.push({ math: math.trim(), display: false });
        return `<span class="katex-placeholder" data-index="${index}"></span>`;
      });

      for (let i = 0; i < codeBlocks.length; i++) {
        content = content.replace(`<!--CODE_BLOCK_PLACEHOLDER_${i}-->`, () => codeBlocks[i]);
      }

      return content;
    });

    hook.doneEach(function () {
      const placeholders = document.querySelectorAll('.katex-placeholder');
      placeholders.forEach(function (el) {
        const index = parseInt(el.getAttribute('data-index'), 10);
        if (isNaN(index) || !mathBlocks[index]) return;
        const mathData = mathBlocks[index];
        try {
          katex.render(mathData.math, el, {
            displayMode: mathData.display,
            throwOnError: false,
          });
        } catch (err) {
          el.textContent = (mathData.display ? '$$' : '$') + mathData.math + (mathData.display ? '$$' : '$');
        }
      });
    });
  }

  window.$docsify = window.$docsify || {};
  window.$docsify.plugins = [].concat(window.$docsify.plugins || [], katexPlugin);
})();
