"""
Simple OCR Diagnostic Tool
Tests if your price region selection and OCR are working correctly
"""
import cv2
import numpy as np
import mss
import pytesseract
import os

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def test_tesseract():
    """Test if Tesseract is installed and accessible"""
    print("\n" + "="*70)
    print("STEP 1: Testing Tesseract Installation")
    print("="*70)
    
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract is installed: Version {version}")
        return True
    except Exception as e:
        print(f"❌ Tesseract NOT found: {e}")
        print("\nPlease install Tesseract:")
        print("  Download: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Install to: C:\\Program Files\\Tesseract-OCR")
        return False

def select_and_test_region():
    """Select region and test OCR on it"""
    print("\n" + "="*70)
    print("STEP 2: Select Your Price Region")
    print("="*70)
    print("\nInstructions:")
    print("  1. A window will show your screen")
    print("  2. Click and drag to select ONLY the price number")
    print("  3. Press SPACE or ENTER to confirm")
    print("  4. Press ESC to cancel")
    print("\nTIPS:")
    print("  - Select a small area around the price (e.g., '21,123.75')")
    print("  - Don't include symbols like $ or +/-")
    print("  - Include a small margin (5-10 pixels)")
    
    input("\nPress ENTER when ready...")
    
    # Capture full screen
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Select region
    print("\n📸 Select the price region now...")
    roi = cv2.selectROI("Select PRICE Region", img, False)
    cv2.destroyAllWindows()
    
    if roi[2] == 0 or roi[3] == 0:
        print("\n❌ No region selected")
        return None
    
    x, y, w, h = roi
    print(f"\n✅ Region selected: x={x}, y={y}, width={w}, height={h}")
    
    # Extract the region
    price_region = img[y:y+h, x:x+w]
    
    # Save the raw region
    cv2.imwrite("test_ocr_raw.png", price_region)
    print(f"💾 Saved raw region to: test_ocr_raw.png")
    
    return price_region

def test_ocr_methods(roi):
    """Test all OCR preprocessing methods"""
    print("\n" + "="*70)
    print("STEP 3: Testing OCR Methods")
    print("="*70)
    
    methods = [
        ("OTSU Threshold", lambda img: cv2.threshold(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]),
        
        ("Adaptive Threshold", lambda img: cv2.adaptiveThreshold(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )),
        
        ("Grayscale Only", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        
        ("Inverted Binary", lambda img: cv2.threshold(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]),
    ]
    
    best_result = None
    best_method = None
    
    for i, (name, preprocess) in enumerate(methods, 1):
        print(f"\n{i}. Testing: {name}")
        try:
            # Preprocess
            processed = preprocess(roi.copy())
            
            # Save processed image
            filename = f"test_ocr_method{i}_{name.replace(' ', '_').lower()}.png"
            cv2.imwrite(filename, processed)
            print(f"   💾 Saved: {filename}")
            
            # Try OCR
            text = pytesseract.image_to_string(
                processed,
                config='--psm 7 -c tessedit_char_whitelist=0123456789.,'
            )
            text = text.strip().replace(',', '').replace(' ', '')
            
            print(f"   📝 Raw OCR output: '{text}'")
            
            # Try to parse as price
            if text:
                try:
                    price = float(text)
                    if 10000 < price < 30000:
                        print(f"   ✅ SUCCESS! Extracted price: {price}")
                        if best_result is None:
                            best_result = price
                            best_method = name
                    else:
                        print(f"   ⚠️  Number {price} outside NQ range (10,000-30,000)")
                except ValueError:
                    print(f"   ❌ Could not convert '{text}' to number")
            else:
                print(f"   ❌ No text detected")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return best_result, best_method

def display_results(best_result, best_method):
    """Display final results"""
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    if best_result:
        print(f"\n✅ OCR WORKING!")
        print(f"   Best method: {best_method}")
        print(f"   Extracted price: {best_result}")
        print("\n🎉 Your price OCR should work in the vision system!")
    else:
        print(f"\n❌ OCR NOT WORKING")
        print("\nPossible issues:")
        print("  1. Region selected doesn't contain just the price")
        print("  2. Price text is too small or blurry")
        print("  3. Poor contrast between text and background")
        print("  4. Extra symbols or text in the region")
        
        print("\nSolutions:")
        print("  1. Look at the saved images (test_ocr_*.png)")
        print("  2. Can you clearly read the price in test_ocr_raw.png?")
        print("  3. Try selecting a different/better region")
        print("  4. Increase TradingView font size")
        print("  5. Use dark theme with white text")
        
        print("\n💡 TIP: The vision system still works without price OCR!")
        print("   Pattern detection and signals work fine, you just won't")
        print("   see the current price in the logs.")
    
    print("\n" + "="*70)
    print("Saved files to check:")
    files = [f for f in os.listdir('.') if f.startswith('test_ocr_')]
    for f in files:
        print(f"  📄 {f}")
    print("="*70)

def main():
    """Main diagnostic flow"""
    print("\n" + "="*70)
    print("PRICE OCR DIAGNOSTIC TOOL")
    print("="*70)
    print("\nThis tool will:")
    print("  1. Check if Tesseract is installed")
    print("  2. Let you select a price region")
    print("  3. Test multiple OCR methods")
    print("  4. Show you what works (or doesn't)")
    print("="*70)
    
    # Test 1: Tesseract
    if not test_tesseract():
        print("\n❌ Cannot continue without Tesseract")
        input("\nPress ENTER to exit...")
        return
    
    # Test 2: Region selection
    roi = select_and_test_region()
    if roi is None:
        print("\n❌ No region selected, exiting...")
        input("\nPress ENTER to exit...")
        return
    
    # Test 3: OCR methods
    best_result, best_method = test_ocr_methods(roi)
    
    # Results
    display_results(best_result, best_method)
    
    input("\nPress ENTER to exit...")

if __name__ == "__main__":
    main()
