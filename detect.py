from ultralytics import YOLO
import cv2
import time

# Carregar modelo treinado
try:
    model = YOLO("runs/detect/epi_detector/weights/best.pt")
    print("✅ Modelo carregado com sucesso!")
except:
    print("❌ Erro ao carregar o modelo. Verifique se o treinamento foi concluído.")
    print("Execute primeiro: python train.py")
    exit()

# Mapeamento de classes
class_names = {
    0: "capacete",
    1: "colete de segurança", 
    2: "luvas de proteção",
    3: "óculos de proteção",
    4: "sem epi"
}

# Cores para cada classe (BGR)
colors = {
    "capacete": (0, 255, 0),           # Verde
    "colete de segurança": (255, 0, 0), # Azul
    "luvas de proteção": (0, 255, 255), # Amarelo
    "óculos de proteção": (255, 0, 255),# Magenta
    "sem epi": (0, 0, 255)              # Vermelho
}

# Inicializar webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Largura
cap.set(4, 720)   # Altura

# Variáveis para FPS
prev_time = 0
new_time = 0

# Contadores para estatísticas
frame_count = 0
detection_stats = {class_name: 0 for class_name in class_names.values()}

print("Iniciando detecção em tempo real...")
print("Pressione 'q' para sair")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Calcular FPS
    new_time = time.time()
    fps = 1 / (new_time - prev_time)
    prev_time = new_time
    
    # Fazer detecção
    results = model(frame, conf=0.5, verbose=False)
    
    # Resetar contadores para este frame
    current_detections = {class_name: 0 for class_name in class_names.values()}
    
    # Processar resultados
    for result in results:
        for box in result.boxes:
            # Extrair coordenadas e classe
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Obter nome da classe e cor
            class_name = class_names[class_id]
            color = colors[class_name]
            
            # Atualizar estatísticas
            current_detections[class_name] += 1
            detection_stats[class_name] += 1
            
            # Desenhar caixa e label
            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Adicionar FPS no frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostrar estatísticas no frame
    y_offset = 70
    for class_name, count in current_detections.items():
        if count > 0:
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[class_name], 2)
            y_offset += 30
    
    # Mostrar frame
    cv2.imshow('EPI Detector - ProtegeAÍ', frame)
    
    # Exibir detecções no terminal a cada 10 frames
    frame_count += 1
    if frame_count % 10 == 0:
        print("\n" + "="*50)
        print(f"Frame {frame_count} - Detecções:")
        for class_name, count in current_detections.items():
            if count > 0:
                print(f"  ✅ {class_name}: {count} detecção(ões)")
        if not any(current_detections.values()):
            print("  ❌ Nenhum objeto detectado")
        print("="*50)
    
    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Estatísticas finais
print("\n" + "="*50)
print("ESTATÍSTICAS FINAIS:")
print("="*50)
total_detections = sum(detection_stats.values())
for class_name, count in detection_stats.items():
    if count > 0:
        percentage = (count / total_detections) * 100 if total_detections > 0 else 0
        print(f"  {class_name}: {count} detecções ({percentage:.1f}%)")

cap.release()
cv2.destroyAllWindows()
print("✅ Detecção encerrada!")