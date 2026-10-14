import sys
import time
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtCore import QUrl

class DebugPage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS [{level}]: {message} at {sourceID}:{lineNumber}")

app = QApplication(sys.argv)
v = QWebEngineView()
page = DebugPage()
v.setPage(page)
v.resize(1100, 750)

# Enable WebGL and local file access
settings = page.settings()
settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

loaded = [False]
def on_load(ok):
    print("Page load finished with status:", ok)
    loaded[0] = True

v.loadFinished.connect(on_load)
html_file = Path("src/lora/ui/droplet.html").resolve()
print("Loading:", html_file)
v.load(QUrl.fromLocalFile(str(html_file)))
v.show()

start = time.time()
while time.time() - start < 4.0:
    app.processEvents()
    time.sleep(0.05)

pix = v.grab()
pix.save("scratch/web_finished.png")
print("Saved screenshot of QWebEngineView!")
app.quit()
