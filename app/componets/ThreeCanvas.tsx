"use client";

import React, {
  useRef,
  useEffect,
  useState,
  forwardRef,
  useImperativeHandle,
  ForwardedRef,
} from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// 定义组件Props和暴露的方法类型
// 在ThreeCanvas.tsx中
interface ThreeCanvasProps {
  ref?: ForwardedRef<ThreeCanvasRef>;
  className?: string;
}

export interface ThreeCanvasRef {
  executeCode: (code: string) => Promise<{
    success: boolean;
    error?: string;
  }>;
}

// 定义资源类型
interface Resources {
  geometries: THREE.BufferGeometry[];
  materials: THREE.Material[];
  objects: THREE.Object3D[];
  animations: THREE.AnimationClip[];
  mixers: THREE.AnimationMixer[];
  updateFunctions: ((delta: number) => void)[];
}

// 定义场景类型
interface SceneRef {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
}

const ThreeCanvas = forwardRef<ThreeCanvasRef, ThreeCanvasProps>(
  function ThreeCanvas(props, ref: ForwardedRef<ThreeCanvasRef>) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const sceneRef = useRef<SceneRef | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);
    const [_lastError, setLastError] = useState<string | null>(null);

    // 创建跟踪资源的对象，用于清理
    const resourcesRef = useRef<Resources>({
      geometries: [],
      materials: [],
      objects: [],
      animations: [],
      mixers: [],
      updateFunctions: [], // 存储场景更新函数
    });

    // 初始化Three.js环境
    useEffect(() => {
      if (!canvasRef.current) return;

      // 创建基本场景
      const canvas = canvasRef.current;
      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
      renderer.setSize(canvas.clientWidth, canvas.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      rendererRef.current = renderer;

      // 初始化默认场景
      const { scene, camera } = initDefaultScene();
      sceneRef.current = { scene, camera };

      // 添加轨道控制器
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controlsRef.current = controls;

      // 渲染循环
      function animate() {
        requestAnimationFrame(animate);
        if (
          sceneRef.current &&
          sceneRef.current.scene &&
          sceneRef.current.camera
        ) {
          if (controlsRef.current) controlsRef.current.update();
          renderer.render(sceneRef.current.scene, sceneRef.current.camera);
        }
      }
      animate();

      // 处理窗口大小变化
      const handleResize = () => {
        if (!canvasRef.current || !sceneRef.current) return;

        const width = canvasRef.current.clientWidth;
        const height = canvasRef.current.clientHeight;

        sceneRef.current.camera.aspect = width / height;
        sceneRef.current.camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      };

      window.addEventListener("resize", handleResize);

      // 清理函数
      return () => {
        window.removeEventListener("resize", handleResize);

        if (controlsRef.current) {
          controlsRef.current.dispose();
          controlsRef.current = null;
        }

        if (renderer) {
          renderer.dispose();
          rendererRef.current = null;
        }

        if (sceneRef.current) {
          disposeScene(sceneRef.current.scene);
          sceneRef.current = null;
        }

        disposeResources();
      };
    }, []);

    // 初始化默认场景
    function initDefaultScene(): SceneRef {
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x111111);

      const camera = new THREE.PerspectiveCamera(
        75,
        canvasRef.current
          ? canvasRef.current.clientWidth / canvasRef.current.clientHeight
          : 1,
        0.1,
        1000
      );
      camera.position.set(0, 0, 5);
      camera.lookAt(0, 0, 0);

      // 添加基本光源
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
      scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(1, 1, 1);
      scene.add(directionalLight);

      // 添加一个默认的网格地面
      const gridHelper = new THREE.GridHelper(10, 10);
      scene.add(gridHelper);

      return { scene, camera };
    }

    // 清理场景资源
    function disposeScene(scene: THREE.Scene): void {
      if (!scene) return;

      scene.traverse((object) => {
        if ((object as THREE.Mesh).geometry) {
          (object as THREE.Mesh).geometry.dispose();
        }

        if ((object as THREE.Mesh).material) {
          if (Array.isArray((object as THREE.Mesh).material)) {
            ((object as THREE.Mesh).material as THREE.Material[]).forEach(
              (material) => material.dispose()
            );
          } else {
            ((object as THREE.Mesh).material as THREE.Material).dispose();
          }
        }
      });
    }

    // 清理资源
    function disposeResources(): void {
      const resources = resourcesRef.current;

      // 清理几何体
      resources.geometries.forEach((geometry) => {
        if (geometry && typeof geometry.dispose === "function") {
          geometry.dispose();
        }
      });

      // 清理材质
      resources.materials.forEach((material) => {
        if (material && typeof material.dispose === "function") {
          material.dispose();
        }
      });

      // 清理动画混合器
      resources.mixers.forEach((mixer) => {
        // 混合器没有dispose方法，但我们可以停止它
        if (mixer && typeof mixer.stopAllAction === "function") {
          mixer.stopAllAction();
        }
      });

      // 重置资源跟踪
      resourcesRef.current = {
        geometries: [],
        materials: [],
        objects: [],
        animations: [],
        mixers: [],
        updateFunctions: [],
      };
    }

    // 定义一个类型来确保沙箱环境中的变量和函数
    interface SandboxEnvironment {
      THREE: typeof THREE;
      scene: THREE.Scene;
      camera: THREE.PerspectiveCamera;
      renderer: THREE.WebGLRenderer;
      canvas: HTMLCanvasElement;
      console: typeof console;
      OrbitControls: typeof OrbitControls;
      trackResource: (resource: unknown, type: string) => unknown;
    }

    // 向Agent暴露的方法
    useImperativeHandle(ref, () => ({
      // 执行Three.js代码 - 直接执行而非通过沙箱
      executeCode: async (code: string) => {
        try {
          // 确保渲染器已初始化
          if (!rendererRef.current) {
            throw new Error("渲染器未初始化");
          }

          // 清除之前的场景对象（除了灯光和网格）
          const scene = sceneRef.current!.scene;
          const objectsToRemove: THREE.Object3D[] = [];

          scene.traverse((object) => {
            // 保留灯光、相机和网格辅助线
            if (
              object !== scene &&
              !(object instanceof THREE.Light) &&
              !(object instanceof THREE.Camera) &&
              !(object instanceof THREE.GridHelper)
            ) {
              objectsToRemove.push(object);
            }
          });

          // 移除对象
          objectsToRemove.forEach((object) => {
            scene.remove(object);
            if ((object as THREE.Mesh).geometry)
              (object as THREE.Mesh).geometry.dispose();
            if ((object as THREE.Mesh).material) {
              if (Array.isArray((object as THREE.Mesh).material)) {
                ((object as THREE.Mesh).material as THREE.Material[]).forEach(
                  (material) => material.dispose()
                );
              } else {
                ((object as THREE.Mesh).material as THREE.Material).dispose();
              }
            }
          });

          // 准备执行环境
          const sandbox: SandboxEnvironment = {
            THREE,
            scene: sceneRef.current!.scene,
            camera: sceneRef.current!.camera,
            renderer: rendererRef.current,
            canvas: canvasRef.current!,
            console: console,
            // 确保OrbitControls可用
            OrbitControls: OrbitControls,
            // 添加资源跟踪函数
            trackResource: (resource, type) => {
              if (!resource) return resource;

              if (
                type === "geometry" &&
                (resource as THREE.BufferGeometry).isBufferGeometry
              ) {
                resourcesRef.current.geometries.push(
                  resource as THREE.BufferGeometry
                );
              } else if (
                type === "material" &&
                (resource as THREE.Material).isMaterial
              ) {
                resourcesRef.current.materials.push(resource as THREE.Material);
              } else if (
                type === "object" &&
                (resource as THREE.Object3D).isObject3D
              ) {
                resourcesRef.current.objects.push(resource as THREE.Object3D);
              } else if (type === "mixer") {
                resourcesRef.current.mixers.push(
                  resource as THREE.AnimationMixer
                );
              }

              return resource;
            },
          };

          // 提取动画函数
          let animationCode = "";
          const animateFunctionMatch = code.match(
            /function\s+animate\s*\(\s*\)\s*{([\s\S]*?)}/i
          );
          if (animateFunctionMatch) {
            // 提取动画函数内部逻辑，移除requestAnimationFrame调用
            const animateBody = animateFunctionMatch[1].replace(
              /requestAnimationFrame\s*\(\s*animate\s*\)\s*;/g,
              ""
            );

            // 创建更新函数代码
            animationCode = `
          // 创建更新函数供ThreeCanvas使用
          const updateFunction = function(delta) {
            ${animateBody}
          };
          // 添加到更新函数列表
          resourcesRef.current.updateFunctions.push(updateFunction);
          `;
          }

          // 包装代码在一个异步函数中执行
          const wrappedCode = `
          (async function() {
            try {
              ${code}
              ${animationCode}
              return { success: true };
            } catch (error) {
              console.error("Three.js代码执行错误:", error);
              return { success: false, error: error.message };
            }
          })();
        `;

          // 执行代码
          const result = await eval(
            `(function(THREE, scene, camera, renderer, canvas, console, trackResource, OrbitControls) {
            return ${wrappedCode};
          })(sandbox.THREE, sandbox.scene, sandbox.camera, sandbox.renderer, sandbox.canvas, sandbox.console, sandbox.trackResource, THREE.OrbitControls)`
          );

          // 清除错误状态
          if (result.success) {
            setLastError(null);
          } else {
            setLastError(result.error);
          }

          return result;
        } catch (error) {
          const err = error as Error;
          setLastError(err.message);
          return {
            success: false,
            error: err.message,
          };
        }
      },
    }));

    return (
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ display: "block" }}
      />
    );
  }
);

export default ThreeCanvas;
