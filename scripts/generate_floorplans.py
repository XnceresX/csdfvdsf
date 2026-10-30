"""
scripts/generate_floorplans.py
Generates high-resolution architectural blueprint images (PNG + SVG)
for the 4 floors of Mayfair Apartment:
- Ground Floor (CAM 1, CAM 2, CAM 3, CAM 4, CAM 6)
- 1st Floor Landing (CAM 5, Flat 1A, Flat 1B)
- 2nd Floor Landing (CAM 7, Flat 2A, Flat 2B)
- Terrace (CAM 8, Terrace Door, Terrace Area)
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def get_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

def draw_blueprint_grid(draw, width, height, step=40):
    for x in range(0, width, step):
        draw.line([(x, 0), (x, height)], fill="#111827", width=1)
    for y in range(0, height, step):
        draw.line([(0, y), (width, y)], fill="#111827", width=1)

def draw_camera_node(draw, x, y, label, angle_deg, color="#10b981", fov_len=70):
    import math
    rad = math.radians(angle_deg)
    # Draw FOV cone
    fov_half = math.radians(35)
    p1 = (x + fov_len * math.cos(rad - fov_half), y + fov_len * math.sin(rad - fov_half))
    p2 = (x + fov_len * math.cos(rad + fov_half), y + fov_len * math.sin(rad + fov_half))
    draw.polygon([(x, y), p1, p2], fill="#10b98122", outline=color)
    
    # Outer circle
    draw.ellipse([x - 18, y - 18, x + 18, y + 18], fill="#1e293b", outline=color, width=2)
    # Inner lens
    draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color)
    # Text badge
    font = get_font(14)
    draw.text((x - 22, y + 22), label, fill="#f8fafc", font=font)

def create_ground_floor(out_png: Path):
    W, H = 1200, 800
    img = Image.new("RGBA", (W, H), "#0a0f1d")
    draw = ImageDraw.Draw(img)
    draw_blueprint_grid(draw, W, H)

    font_title = get_font(24)
    font_sub = get_font(14)
    font_zone = get_font(13)

    # Header
    draw.text((40, 30), "MAYFAIR APARTMENT - GROUND FLOOR PLAN", fill="#38bdf8", font=font_title)
    draw.text((40, 65), "SURVEILLANCE TOPOLOGY & MONITORING ZONES", fill="#94a3b8", font=font_sub)

    # Perimeter walls
    draw.rectangle([100, 110, 1100, 720], outline="#475569", width=4, fill="#0f172a")

    # Internal layout: Parking bays, walkways, rooms
    # North Walkway (top)
    draw.rectangle([110, 120, 1090, 220], outline="#334155", width=2, fill="#131e36")
    draw.text((500, 160), "NORTH BACK WALKWAY (CAM 6)", fill="#94a3b8", font=font_zone)

    # East Walkway (right)
    draw.rectangle([880, 230, 1090, 620], outline="#334155", width=2, fill="#131e36")
    draw.text((910, 410), "EAST WALKWAY\n(CAM 1)", fill="#94a3b8", font=font_zone)

    # Parking Left
    draw.rectangle([120, 240, 480, 620], outline="#334155", width=2, fill="#0e1726")
    draw.text((220, 400), "GROUND PARKING LEFT (CAM 4)", fill="#94a3b8", font=font_zone)
    for y in range(290, 600, 70):
        draw.line([(130, y), (280, y)], fill="#1e293b", width=2)
        draw.text((140, y - 20), "BAY", fill="#334155", font=get_font(10))

    # Parking Right
    draw.rectangle([500, 240, 860, 620], outline="#334155", width=2, fill="#0e1726")
    draw.text((600, 400), "GROUND PARKING RIGHT (CAM 3)", fill="#94a3b8", font=font_zone)
    for y in range(290, 600, 70):
        draw.line([(700, y), (850, y)], fill="#1e293b", width=2)
        draw.text((810, y - 20), "BAY", fill="#334155", font=get_font(10))

    # Ground Stairwell core (between left and right parking)
    draw.rectangle([460, 240, 520, 380], outline="#f59e0b", width=2, fill="#1e1b4b")
    draw.text((465, 300), "STAIRS\nTO 1F", fill="#fcd34d", font=get_font(11))

    # SE Corner / Gate area
    draw.rectangle([780, 630, 1090, 710], outline="#ef4444", width=2, fill="#1f1a24")
    draw.text((850, 660), "MAIN GATE / SE CORNER (CAM 2)", fill="#fca5a5", font=font_zone)

    # Store Room
    draw.rectangle([120, 630, 320, 710], outline="#64748b", width=2, fill="#1e293b")
    draw.text((150, 660), "ROOM / STORE", fill="#cbd5e1", font=font_zone)

    # Draw Cameras
    # CAM 1: Ground Floor East Walkway
    draw_camera_node(draw, 1000, 330, "CAM 1", 180)
    # CAM 2: Ground Floor SE Corner / Gate
    draw_camera_node(draw, 960, 670, "CAM 2", 225)
    # CAM 3: Ground Floor Parking Right
    draw_camera_node(draw, 740, 320, "CAM 3", 210)
    # CAM 4: Ground Floor Parking Left
    draw_camera_node(draw, 340, 320, "CAM 4", 330)
    # CAM 6: Ground Floor North Back Walkway
    draw_camera_node(draw, 620, 160, "CAM 6", 180)

    # Legend / Floor indicator
    draw.rectangle([980, 40, 1150, 80], outline="#38bdf8", fill="#0369a133", width=2)
    draw.text((1000, 50), "FLOOR 0 (GROUND)", fill="#38bdf8", font=get_font(14))

    img.save(out_png)

def create_floor1(out_png: Path):
    W, H = 1200, 800
    img = Image.new("RGBA", (W, H), "#0a0f1d")
    draw = ImageDraw.Draw(img)
    draw_blueprint_grid(draw, W, H)

    font_title = get_font(24)
    font_sub = get_font(14)
    font_zone = get_font(14)

    # Header
    draw.text((40, 30), "MAYFAIR APARTMENT - 1ST FLOOR LANDING", fill="#38bdf8", font=font_title)
    draw.text((40, 65), "RESIDENTIAL FLATS & STAIRWELL CORRIDOR", fill="#94a3b8", font=font_sub)

    # Perimeter walls
    draw.rectangle([180, 140, 1020, 700], outline="#475569", width=4, fill="#0f172a")

    # Flat 1A (Left Wing)
    draw.rectangle([200, 160, 500, 680], outline="#38bdf8", width=2, fill="#0c1e3d")
    draw.text((290, 390), "FLAT 1A\n(RESIDENTIAL)", fill="#7dd3fc", font=font_zone)
    # Door 1A
    draw.rectangle([485, 360, 505, 440], fill="#f59e0b", outline="#fbbf24")
    draw.text((440, 450), "DOOR 1A", fill="#fbbf24", font=get_font(12))

    # Flat 1B (Right Wing)
    draw.rectangle([700, 160, 1000, 680], outline="#38bdf8", width=2, fill="#0c1e3d")
    draw.text((790, 390), "FLAT 1B\n(RESIDENTIAL)", fill="#7dd3fc", font=font_zone)
    # Door 1B
    draw.rectangle([695, 360, 715, 440], fill="#f59e0b", outline="#fbbf24")
    draw.text((720, 450), "DOOR 1B", fill="#fbbf24", font=get_font(12))

    # Central Corridor & Stairwell Landing
    draw.rectangle([515, 160, 685, 680], outline="#334155", width=2, fill="#131e36")
    draw.text((530, 200), "1ST FLOOR LANDING\n(CAM 5)", fill="#e2e8f0", font=font_zone)

    # Stairs Down (to Ground)
    draw.rectangle([540, 520, 660, 660], outline="#f59e0b", width=2, fill="#1e1b4b")
    draw.text((550, 580), "STAIRS DOWN\n(TO GROUND)", fill="#fcd34d", font=get_font(12))

    # Stairs Up (to Floor 2)
    draw.rectangle([540, 260, 660, 380], outline="#f59e0b", width=2, fill="#1e1b4b")
    draw.text((555, 310), "STAIRS UP\n(TO 2F)", fill="#fcd34d", font=get_font(12))

    # CAM 5: First Floor Stairwell Landing
    draw_camera_node(draw, 600, 440, "CAM 5", 90)

    # Floor Badge
    draw.rectangle([980, 40, 1150, 80], outline="#38bdf8", fill="#0369a133", width=2)
    draw.text((1010, 50), "FLOOR 1 (1F)", fill="#38bdf8", font=get_font(14))

    img.save(out_png)

def create_floor2(out_png: Path):
    W, H = 1200, 800
    img = Image.new("RGBA", (W, H), "#0a0f1d")
    draw = ImageDraw.Draw(img)
    draw_blueprint_grid(draw, W, H)

    font_title = get_font(24)
    font_sub = get_font(14)
    font_zone = get_font(14)

    # Header
    draw.text((40, 30), "MAYFAIR APARTMENT - 2ND FLOOR LANDING", fill="#38bdf8", font=font_title)
    draw.text((40, 65), "UPPER APARTMENTS & TERRACE ACCESS CORRIDOR", fill="#94a3b8", font=font_sub)

    # Perimeter walls
    draw.rectangle([180, 140, 1020, 700], outline="#475569", width=4, fill="#0f172a")

    # Flat 2A (Left Wing)
    draw.rectangle([200, 160, 500, 680], outline="#a855f7", width=2, fill="#1e1236")
    draw.text((270, 390), "FLAT 2A (LEFT)\n(RESIDENTIAL)", fill="#d8b4fe", font=font_zone)
    # Door 2A
    draw.rectangle([485, 360, 505, 440], fill="#f59e0b", outline="#fbbf24")
    draw.text((420, 450), "DOOR 2A", fill="#fbbf24", font=get_font(12))

    # Flat 2B (Right Wing)
    draw.rectangle([700, 160, 1000, 680], outline="#a855f7", width=2, fill="#1e1236")
    draw.text((770, 390), "FLAT 2B (RIGHT)\n(RESIDENTIAL)", fill="#d8b4fe", font=font_zone)
    # Door 2B
    draw.rectangle([695, 360, 715, 440], fill="#f59e0b", outline="#fbbf24")
    draw.text((720, 450), "DOOR 2B", fill="#fbbf24", font=get_font(12))

    # Central Corridor & Stairwell Landing
    draw.rectangle([515, 160, 685, 680], outline="#334155", width=2, fill="#131e36")
    draw.text((530, 200), "2ND FLOOR LANDING\n(CAM 7)", fill="#e2e8f0", font=font_zone)

    # Stairs Down (to 1F)
    draw.rectangle([540, 520, 660, 660], outline="#f59e0b", width=2, fill="#1e1b4b")
    draw.text((560, 580), "STAIRS DOWN\n(TO 1F)", fill="#fcd34d", font=get_font(12))

    # Stairs Up (to Terrace)
    draw.rectangle([540, 260, 660, 380], outline="#f59e0b", width=2, fill="#1e1b4b")
    draw.text((550, 310), "STAIRS UP\n(TO TERRACE)", fill="#fcd34d", font=get_font(12))

    # CAM 7: Second Floor Stairwell Landing
    draw_camera_node(draw, 600, 440, "CAM 7", 90)

    # Floor Badge
    draw.rectangle([980, 40, 1150, 80], outline="#a855f7", fill="#7e22ce33", width=2)
    draw.text((1010, 50), "FLOOR 2 (2F)", fill="#d8b4fe", font=get_font(14))

    img.save(out_png)

def create_terrace(out_png: Path):
    W, H = 1200, 800
    img = Image.new("RGBA", (W, H), "#0a0f1d")
    draw = ImageDraw.Draw(img)
    draw_blueprint_grid(draw, W, H)

    font_title = get_font(24)
    font_sub = get_font(14)
    font_zone = get_font(14)

    # Header
    draw.text((40, 30), "MAYFAIR APARTMENT - TOP OF BUILDING / TERRACE", fill="#38bdf8", font=font_title)
    draw.text((40, 65), "OPEN ROOFTOP, TERRACE DOOR & UTILITY SPACES", fill="#94a3b8", font=font_sub)

    # Perimeter walls (parapet)
    draw.rectangle([140, 130, 1060, 710], outline="#06b6d4", width=4, fill="#0c192c")

    # Terrace open area
    draw.text((250, 380), "OPEN TERRACE AREA (CAM 8)", fill="#67e8f9", font=font_zone)

    # Water tank & solar utility block
    draw.rectangle([180, 160, 400, 300], outline="#0284c7", width=2, fill="#07598533")
    draw.text((210, 220), "WATER TANK & UTILITY", fill="#38bdf8", font=get_font(12))

    # Solar Array Area
    draw.rectangle([180, 520, 520, 680], outline="#0284c7", width=2, fill="#07598533")
    draw.text((260, 590), "SOLAR PANEL INSTALLATION", fill="#38bdf8", font=get_font(12))

    # Stairwell Headhouse & Terrace Door
    draw.rectangle([720, 280, 950, 520], outline="#f59e0b", width=3, fill="#1c1917")
    draw.text((750, 320), "STAIRWELL HEADHOUSE", fill="#fdba74", font=get_font(13))
    # Terrace Door
    draw.rectangle([705, 370, 725, 450], fill="#ef4444", outline="#f87171", width=2)
    draw.text((735, 400), "TERRACE DOOR\n(ACCESS POINT)", fill="#fca5a5", font=get_font(12))

    # Stairs going down
    draw.rectangle([760, 440, 910, 500], outline="#d97706", fill="#451a03")
    draw.text((780, 460), "STAIRS TO 2F", fill="#fde68a", font=get_font(11))

    # CAM 8: Top of Building / Terrace
    draw_camera_node(draw, 640, 410, "CAM 8", 180)

    # Floor Badge
    draw.rectangle([980, 40, 1150, 80], outline="#06b6d4", fill="#0891b233", width=2)
    draw.text((1020, 50), "TERRACE", fill="#67e8f9", font=get_font(14))

    img.save(out_png)

def main():
    out_dir = Path("data/mayfair/floorplans")
    out_dir.mkdir(parents=True, exist_ok=True)
    web_dir = Path("web/static/maps")
    web_dir.mkdir(parents=True, exist_ok=True)

    create_ground_floor(out_dir / "ground_floor.png")
    create_floor1(out_dir / "floor1.png")
    create_floor2(out_dir / "floor2.png")
    create_terrace(out_dir / "terrace.png")

    import shutil
    for fname in ["ground_floor.png", "floor1.png", "floor2.png", "terrace.png"]:
        shutil.copy(out_dir / fname, web_dir / fname)
    print("Floor plans generated successfully in data/mayfair/floorplans and web/static/maps!")

if __name__ == "__main__":
    main()
