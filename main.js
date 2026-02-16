(function(){
  const output = document.getElementById('output')
  const status = document.getElementById('status')
  let es = null

  function appendWord(word, isStart){
    if(isStart){
      const p = document.createElement('div')
      p.className = 'line'
      output.appendChild(p)
      if(word) p.appendChild(makeWordNode(word))
      // scroll into view
      p.scrollIntoView({behavior:'smooth', block:'end'})
    } else {
      const line = output.lastElementChild || (function(){ let p = document.createElement('div'); p.className='line'; output.appendChild(p); return p })()
      line.appendChild(makeWordNode(word))
      line.scrollIntoView({behavior:'smooth', block:'end'})
    }
  }

  function makeWordNode(word){
    const span = document.createElement('span')
    span.className = 'word'
    span.textContent = word
    return span
  }

  document.getElementById('start').addEventListener('click', ()=>{
    const cfg = document.getElementById('config')
    const file = cfg && cfg.dataset ? cfg.dataset.file : ''
    const delay = cfg && cfg.dataset ? cfg.dataset.delay : ''
    if(es){ es.close(); es = null }
    output.innerHTML = ''
    status.textContent = 'Connecting...'
    // If file/delay are present, pass them as query params; otherwise let server defaults apply
    const params = []
    if(file) params.push(`file=${encodeURIComponent(file)}`)
    if(delay) params.push(`delay=${encodeURIComponent(delay)}`)
    const url = `/stream${params.length ? ('?' + params.join('&')) : ''}`
    es = new EventSource(url)
    es.onmessage = function(e){
      try{
        const obj = JSON.parse(e.data)
        if(obj.event === 'end'){
          status.textContent = 'Completed'
          es.close(); es = null
          return
        }
        const word = obj.word
        const is_start = obj.is_start
        if(word === ""){
          // blank line
          appendWord('', true)
        } else {
          appendWord(word, is_start)
        }
      }catch(err){
        console.error('Invalid message', e.data)
      }
    }
    es.onerror = function(){ status.textContent = 'Connection error or closed'; }
  })

  document.getElementById('stop').addEventListener('click', ()=>{
    if(es){ es.close(); es = null; status.textContent = 'Stopped by user' }
  })
})();